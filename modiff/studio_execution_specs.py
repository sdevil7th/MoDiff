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
FLUX_SCHNELL_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model-00001-of-00002.safetensors",
    "text_encoder_2/model-00002-of-00002.safetensors",
    "text_encoder_2/model.safetensors.index.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/spiece.model",
    "tokenizer_2/tokenizer.json",
    "tokenizer_2/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
FLUX_DEV_REPO = "black-forest-labs/FLUX.1-dev"
FLUX_DEV_DIFFUSERS_FILES = [
    ".gitattributes",
    "LICENSE.md",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model-00001-of-00002.safetensors",
    "text_encoder_2/model-00002-of-00002.safetensors",
    "text_encoder_2/model.safetensors.index.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/spiece.model",
    "tokenizer_2/tokenizer.json",
    "tokenizer_2/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
FLUX_DEV_FP8_REPO = "black-forest-labs/FLUX.1-dev-FP8"
FLUX_KREA_REPO = "black-forest-labs/FLUX.1-Krea-dev"
FLUX_KREA_DIFFUSERS_FILES = [
    ".gitattributes",
    "LICENSE.md",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model-00001-of-00002.safetensors",
    "text_encoder_2/model-00002-of-00002.safetensors",
    "text_encoder_2/model.safetensors.index.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/spiece.model",
    "tokenizer_2/tokenizer.json",
    "tokenizer_2/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
FLUX_DEPTH_REPO = "black-forest-labs/FLUX.1-Depth-dev"
FLUX_CANNY_REPO = "black-forest-labs/FLUX.1-Canny-dev"
FLUX_CONTROL_DIFFUSERS_FILES = [
    ".gitattributes",
    "LICENSE.md",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model-00001-of-00004.safetensors",
    "text_encoder_2/model-00002-of-00004.safetensors",
    "text_encoder_2/model-00003-of-00004.safetensors",
    "text_encoder_2/model-00004-of-00004.safetensors",
    "text_encoder_2/model.safetensors.index.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/spiece.model",
    "tokenizer_2/tokenizer.json",
    "tokenizer_2/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
FLUX_CANNY_VERIFIED_REPAIR_REPO = "fuliucansheng/FLUX.1-Canny-dev-diffusers"
FLUX_REDUX_REPO = "black-forest-labs/FLUX.1-Redux-dev"
FLUX_REDUX_DIFFUSERS_FILES = [
    ".gitattributes",
    "LICENSE.md",
    "README.md",
    "feature_extractor/preprocessor_config.json",
    "image_embedder/config.json",
    "image_embedder/diffusion_pytorch_model.safetensors",
    "image_encoder/config.json",
    "image_encoder/model.safetensors",
    "model_index.json",
]
FLUX_KONTEXT_REPO = "black-forest-labs/FLUX.1-Kontext-dev"
FLUX_KONTEXT_DIFFUSERS_FILES = list(FLUX_DEV_DIFFUSERS_FILES)
FLUX_KONTEXT_NVFP4_REPO = "black-forest-labs/FLUX.1-Kontext-dev-NVFP4"
FLUX_FILL_REPO = "black-forest-labs/FLUX.1-Fill-dev"
FLUX_FILL_DIFFUSERS_FILES = list(FLUX_DEV_DIFFUSERS_FILES)
FLUX2_DEV_REPO = "black-forest-labs/FLUX.2-dev"
FLUX2_DEV_DIFFUSERS_FILES = [
    ".gitattributes",
    "LICENSE.md",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    *[f"text_encoder/model-{index:05d}-of-00010.safetensors" for index in range(1, 11)],
    "text_encoder/model.safetensors.index.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/preprocessor_config.json",
    "tokenizer/processor_config.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    *[
        f"transformer/diffusion_pytorch_model-{index:05d}-of-00007.safetensors"
        for index in range(1, 8)
    ],
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
FLUX2_KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
FLUX2_KLEIN_DIFFUSERS_FILES = [
    ".gitattributes",
    "LICENSE.md",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/added_tokens.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
FLUX2_KLEIN_BASE_REPO = "black-forest-labs/FLUX.2-klein-base-4B"
# The Base and distilled repositories publish the same component-oriented
# Diffusers layout. Keep a separate selection so revisions and capabilities
# never alias merely because their current path sets overlap.
FLUX2_KLEIN_BASE_DIFFUSERS_FILES = list(FLUX2_KLEIN_DIFFUSERS_FILES)
SDXL_BASE_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_TURBO_REPO = "stabilityai/sdxl-turbo"
SDXL_INSTRUCT_PIX2PIX_REPO = "diffusers/sdxl-instructpix2pix-768"
SDXL_BASE_FP16_DIFFUSERS_FILES = [
    ".gitattributes",
    "LICENSE.md",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]
SDXL_TURBO_FP16_DIFFUSERS_FILES = list(SDXL_BASE_FP16_DIFFUSERS_FILES)
SDXL_INSTRUCT_PIX2PIX_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
SDXL_CONTROLNET_CANNY_REPO = "diffusers/controlnet-canny-sdxl-1.0"
SDXL_CONTROLNET_CANNY_FP16_FILES = [
    ".gitattributes",
    "README.md",
    "config.json",
    "diffusion_pytorch_model.fp16.safetensors",
]
SDXL_CONTROLNET_UNION_REPO = "xinsir/controlnet-union-sdxl-1.0"
SDXL_CONTROLNET_UNION_FILES = [
    "config.json",
    "diffusion_pytorch_model.safetensors",
]
SDXL_IP_ADAPTER_REPO = "h94/IP-Adapter"
SDXL_IP_ADAPTER_FILES = [
    "sdxl_models/ip-adapter_sdxl.safetensors",
    "sdxl_models/image_encoder/config.json",
    "sdxl_models/image_encoder/model.safetensors",
]
SDXL_T2I_ADAPTER_CANNY_REPO = "TencentARC/t2i-adapter-canny-sdxl-1.0"
SDXL_T2I_ADAPTER_CANNY_FP16_FILES = [
    ".gitattributes",
    "README.md",
    "config.json",
    "diffusion_pytorch_model.fp16.safetensors",
]
HUNYUAN_DIT_DISTILLED_REPO = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled"
HUNYUAN_DIT_DISTILLED_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model-00001-of-00002.safetensors",
    "text_encoder_2/model-00002-of-00002.safetensors",
    "text_encoder_2/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/spiece.model",
    "tokenizer_2/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
HUNYUAN_DIT_CONTROLNET_CANNY_REPO = "Tencent-Hunyuan/HunyuanDiT-v1.2-ControlNet-Diffusers-Canny"
HUNYUAN_DIT_CONTROLNET_CANNY_FILES = [
    ".gitattributes",
    "README.md",
    "config.json",
    "diffusion_pytorch_model.safetensors",
]
SD15_BASE_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD15_SHARED_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "feature_extractor/preprocessor_config.json",
    "model_index.json",
    "safety_checker/config.json",
    "safety_checker/model.fp16.safetensors",
    "safety_checker/model.safetensors",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder/model.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
]
SD15_CONTROLNET_CANNY_REPO = "lllyasviel/control_v11p_sd15_canny"
SD15_CONTROLNET_CANNY_FILES = [
    ".gitattributes",
    "README.md",
    "config.json",
    "diffusion_pytorch_model.safetensors",
]
SANA_REPO = "Efficient-Large-Model/Sana_600M_1024px_diffusers"
SANA_600M_FP16_DIFFUSERS_FILES = [
    ".gitattributes",
    "LICENSE",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16-00001-of-00002.safetensors",
    "text_encoder/model.fp16-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.fp16.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer.model",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]
SANA_SPRINT_REPO = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
SANA_SPRINT_DIFFUSERS_FILES = [
    ".gitattributes",
    "LICENSE",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer.model",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
PIXART_SIGMA_REPO = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
PIXART_SIGMA_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
KANDINSKY3_REPO = "kandinsky-community/kandinsky-3"
KANDINSKY3_FP16_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "movq/config.json",
    "movq/diffusion_pytorch_model.fp16.safetensors",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16-00001-of-00005.safetensors",
    "text_encoder/model.fp16-00002-of-00005.safetensors",
    "text_encoder/model.fp16-00003-of-00005.safetensors",
    "text_encoder/model.fp16-00004-of-00005.safetensors",
    "text_encoder/model.fp16-00005-of-00005.safetensors",
    "text_encoder/model.safetensors.index.fp16.json",
    "tokenizer/added_tokens.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer_config.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
]
LONGCAT_IMAGE_REPO = "meituan-longcat/LongCat-Image"
LONGCAT_IMAGE_EDIT_REPO = "meituan-longcat/LongCat-Image-Edit"
LONGCAT_IMAGE_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model-00001-of-00005.safetensors",
    "text_encoder/model-00002-of-00005.safetensors",
    "text_encoder/model-00003-of-00005.safetensors",
    "text_encoder/model-00004-of-00005.safetensors",
    "text_encoder/model-00005-of-00005.safetensors",
    "text_encoder/model.safetensors.index.json",
    "text_processor/chat_template.json",
    "text_processor/config.json",
    "text_processor/merges.txt",
    "text_processor/preprocessor_config.json",
    "text_processor/special_tokens_map.json",
    "text_processor/tokenizer.json",
    "text_processor/tokenizer_config.json",
    "text_processor/vocab.json",
    "tokenizer/chat_template.json",
    "tokenizer/config.json",
    "tokenizer/merges.txt",
    "tokenizer/preprocessor_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
LONGCAT_IMAGE_EDIT_DIFFUSERS_FILES = sorted([*LONGCAT_IMAGE_DIFFUSERS_FILES, "text_encoder/preprocessor_config.json"])
LUMINA_REPO = "Alpha-VLLM/Lumina-Next-SFT-diffusers"
LUMINA_NEXT_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer.model",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
LUMINA2_REPO = "Alpha-VLLM/Lumina-Image-2.0"
OMNIGEN_REPO = "Shitao/OmniGen-v1-diffusers"
OMNIGEN_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
OVIS_IMAGE_REPO = "ATH-MaaS/Ovis-Image-7B"
PRX_REPO = "Photoroom/prx-512-t2i-sft"
PRX_DIFFUSERS_FILES = [
    ".gitattributes",
    "LICENSE",
    "NOTICE",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00003.safetensors",
    "text_encoder/model-00002-of-00003.safetensors",
    "text_encoder/model-00003-of-00003.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer.model",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
NUCLEUS_IMAGE_REPO = "NucleusAI/Nucleus-Image"
NUCLEUS_IMAGE_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "processor/chat_template.json",
    "processor/config.json",
    "processor/merges.txt",
    "processor/preprocessor_config.json",
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "processor/video_preprocessor_config.json",
    "processor/vocab.json",
    "scheduler/scheduler_config.json",
    "text_encoder/README.md",
    "text_encoder/chat_template.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/merges.txt",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "text_encoder/model.safetensors.index.json",
    "text_encoder/preprocessor_config.json",
    "text_encoder/tokenizer.json",
    "text_encoder/tokenizer_config.json",
    "text_encoder/video_preprocessor_config.json",
    "text_encoder/vocab.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00004-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00005-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00006-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00007-of-00007.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
OVIS_IMAGE_DIFFUSERS_FILES = [
    "LICENSE",
    "NOTICE",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/added_tokens.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
LUMINA2_DIFFUSERS_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00003.safetensors",
    "text_encoder/model-00002-of-00003.safetensors",
    "text_encoder/model-00003-of-00003.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer.model",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
AURAFLOW_V03_REPO = "fal/AuraFlow-v0.3"
AURAFLOW_V03_DIFFUSERS_FILES = [
    ".gitattributes",
    "LICENSE",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "tokenizer/added_tokens.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer.model",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.fp16.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.fp16.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.fp16.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]
CHROMA1_HD_REPO = "lodestones/Chroma1-HD"
CHROMA1_HD_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/added_tokens.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
COGVIEW3_PLUS_REPO = "zai-org/CogView3-Plus-3B"
COGVIEW3_PLUS_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/added_tokens.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
COGVIEW4_6B_REPO = "zai-org/CogView4-6B"
COGVIEW4_6B_DIFFUSERS_FILES = [
    ".gitattributes",
    "LICENSE",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
ERNIE_IMAGE_TURBO_REPO = "baidu/ERNIE-Image-Turbo"
ERNIE_IMAGE_TURBO_DIFFUSERS_FILES = [
    ".gitattributes",
    "LICENSE",
    "README.md",
    "model_index.json",
    "pe/chat_template.jinja",
    "pe/config.json",
    "pe/generation_config.json",
    "pe/model.safetensors",
    "pe/tokenizer.json",
    "pe/tokenizer_config.json",
    "pe_tokenizer/chat_template.jinja",
    "pe_tokenizer/tokenizer.json",
    "pe_tokenizer/tokenizer_config.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
GLM_IMAGE_REPO = "zai-org/GLM-Image"
GLM_IMAGE_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "processor/chat_template.jinja",
    "processor/preprocessor_config.json",
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/added_tokens.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "vision_language_encoder/config.json",
    "vision_language_encoder/generation_config.json",
    "vision_language_encoder/model-00001-of-00004.safetensors",
    "vision_language_encoder/model-00002-of-00004.safetensors",
    "vision_language_encoder/model-00003-of-00004.safetensors",
    "vision_language_encoder/model-00004-of-00004.safetensors",
    "vision_language_encoder/model.safetensors.index.json",
]
JOYIMAGE_EDIT_REPO = "jdopensource/JoyAI-Image-Edit-Diffusers"
JOYIMAGE_EDIT_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "processor/added_tokens.json",
    "processor/chat_template.jinja",
    "processor/merges.txt",
    "processor/preprocessor_config.json",
    "processor/special_tokens_map.json",
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "processor/video_preprocessor_config.json",
    "processor/vocab.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/added_tokens.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00004-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00005-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00006-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00007-of-00007.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
JOYIMAGE_EDIT_PLUS_REPO = "jdopensource/JoyAI-Image-Edit-Plus-Diffusers"
JOYIMAGE_EDIT_PLUS_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "processor/chat_template.jinja",
    "processor/processor_config.json",
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00004-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00005-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00006-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00007-of-00007.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
DREAMLITE_BASE_REPO = "carlofkl/DreamLite-base"
DREAMLITE_MOBILE_REPO = "carlofkl/DreamLite-mobile"
DREAMLITE_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "processor/added_tokens.json",
    "processor/chat_template.jinja",
    "processor/merges.txt",
    "processor/preprocessor_config.json",
    "processor/special_tokens_map.json",
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "processor/video_preprocessor_config.json",
    "processor/vocab.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model.safetensors",
    "tokenizer/added_tokens.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
LCM_DREAMSHAPER_REPO = "SimianLuo/LCM_Dreamshaper_v7"
LCM_DREAMSHAPER_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "feature_extractor/preprocessor_config.json",
    "model_index.json",
    "safety_checker/config.json",
    "safety_checker/model.safetensors",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
MARIGOLD_DEPTH_LCM_REPO = "prs-eth/marigold-depth-lcm-v1-0"
MARIGOLD_DEPTH_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
SMOLLM2_135M_INSTRUCT_REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"
SMOLLM2_135M_INSTRUCT_TRANSFORMERS_FILES = [
    ".gitattributes",
    "README.md",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]
SMOLVLM_256M_INSTRUCT_REPO = "HuggingFaceTB/SmolVLM-256M-Instruct"
SMOLVLM_256M_INSTRUCT_TRANSFORMERS_FILES = [
    ".gitattributes",
    "README.md",
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]
JANUS_PRO_1B_REPO = "deepseek-community/Janus-Pro-1B"
JANUS_PRO_1B_TRANSFORMERS_FILES = [
    ".gitattributes",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]
WHISPER_TINY_REPO = "openai/whisper-tiny"
WHISPER_TINY_TRANSFORMERS_FILES = [
    ".gitattributes",
    "README.md",
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "normalizer.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]
WAV2VEC2_BASE_960H_REPO = "facebook/wav2vec2-base-960h"
WAV2VEC2_BASE_960H_TRANSFORMERS_FILES = [
    ".gitattributes",
    "README.md",
    "config.json",
    "feature_extractor_config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
]
WAN_22_I2V_A14B_REPO = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
WAN_22_I2V_A14B_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00003.safetensors",
    "text_encoder/model-00002-of-00003.safetensors",
    "text_encoder/model-00003-of-00003.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00012.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00012.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00012.safetensors",
    "transformer/diffusion_pytorch_model-00004-of-00012.safetensors",
    "transformer/diffusion_pytorch_model-00005-of-00012.safetensors",
    "transformer/diffusion_pytorch_model-00006-of-00012.safetensors",
    "transformer/diffusion_pytorch_model-00007-of-00012.safetensors",
    "transformer/diffusion_pytorch_model-00008-of-00012.safetensors",
    "transformer/diffusion_pytorch_model-00009-of-00012.safetensors",
    "transformer/diffusion_pytorch_model-00010-of-00012.safetensors",
    "transformer/diffusion_pytorch_model-00011-of-00012.safetensors",
    "transformer/diffusion_pytorch_model-00012-of-00012.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "transformer_2/config.json",
    "transformer_2/diffusion_pytorch_model-00001-of-00012.safetensors",
    "transformer_2/diffusion_pytorch_model-00002-of-00012.safetensors",
    "transformer_2/diffusion_pytorch_model-00003-of-00012.safetensors",
    "transformer_2/diffusion_pytorch_model-00004-of-00012.safetensors",
    "transformer_2/diffusion_pytorch_model-00005-of-00012.safetensors",
    "transformer_2/diffusion_pytorch_model-00006-of-00012.safetensors",
    "transformer_2/diffusion_pytorch_model-00007-of-00012.safetensors",
    "transformer_2/diffusion_pytorch_model-00008-of-00012.safetensors",
    "transformer_2/diffusion_pytorch_model-00009-of-00012.safetensors",
    "transformer_2/diffusion_pytorch_model-00010-of-00012.safetensors",
    "transformer_2/diffusion_pytorch_model-00011-of-00012.safetensors",
    "transformer_2/diffusion_pytorch_model-00012-of-00012.safetensors",
    "transformer_2/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
WAN_22_T2V_A14B_DIFFUSERS_FILES = WAN_22_I2V_A14B_DIFFUSERS_FILES
WAN_22_TI2V_5B_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN_22_TI2V_5B_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00003.safetensors",
    "text_encoder/model-00002-of-00003.safetensors",
    "text_encoder/model-00003-of-00003.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00005.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
    "transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
    "transformer/diffusion_pytorch_model-00005-of-00005.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
WAN_T2V_1_3B_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN_I2V_14B_480P_REPO = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
WAN_I2V_14B_480P_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "image_encoder/config.json",
    "image_encoder/model.safetensors",
    "image_processor/preprocessor_config.json",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    *[f"text_encoder/model-{index:05d}-of-00005.safetensors" for index in range(1, 6)],
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    *[
        f"transformer/diffusion_pytorch_model-{index:05d}-of-00014.safetensors"
        for index in range(1, 15)
    ],
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
WAN_T2V_1_3B_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00005.safetensors",
    "text_encoder/model-00002-of-00005.safetensors",
    "text_encoder/model-00003-of-00005.safetensors",
    "text_encoder/model-00004-of-00005.safetensors",
    "text_encoder/model-00005-of-00005.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
WAN_VACE_1_3B_REPO = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
WAN_VACE_1_3B_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00003.safetensors",
    "text_encoder/model-00002-of-00003.safetensors",
    "text_encoder/model-00003-of-00003.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
WAN_FLF_14B_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "image_encoder/config.json",
    "image_encoder/model.safetensors",
    "image_processor/merges.txt",
    "image_processor/preprocessor_config.json",
    "image_processor/special_tokens_map.json",
    "image_processor/tokenizer.json",
    "image_processor/tokenizer_config.json",
    "image_processor/vocab.json",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00005.safetensors",
    "text_encoder/model-00002-of-00005.safetensors",
    "text_encoder/model-00003-of-00005.safetensors",
    "text_encoder/model-00004-of-00005.safetensors",
    "text_encoder/model-00005-of-00005.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00014.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00014.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00014.safetensors",
    "transformer/diffusion_pytorch_model-00004-of-00014.safetensors",
    "transformer/diffusion_pytorch_model-00005-of-00014.safetensors",
    "transformer/diffusion_pytorch_model-00006-of-00014.safetensors",
    "transformer/diffusion_pytorch_model-00007-of-00014.safetensors",
    "transformer/diffusion_pytorch_model-00008-of-00014.safetensors",
    "transformer/diffusion_pytorch_model-00009-of-00014.safetensors",
    "transformer/diffusion_pytorch_model-00010-of-00014.safetensors",
    "transformer/diffusion_pytorch_model-00011-of-00014.safetensors",
    "transformer/diffusion_pytorch_model-00012-of-00014.safetensors",
    "transformer/diffusion_pytorch_model-00013-of-00014.safetensors",
    "transformer/diffusion_pytorch_model-00014-of-00014.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
LTX_VIDEO_REPO = "Lightricks/LTX-Video-0.9.8-13B-distilled"
LTX_VIDEO_DIFFUSERS_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/added_tokens.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00006.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00006.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00006.safetensors",
    "transformer/diffusion_pytorch_model-00004-of-00006.safetensors",
    "transformer/diffusion_pytorch_model-00005-of-00006.safetensors",
    "transformer/diffusion_pytorch_model-00006-of-00006.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    # Revision 7c64400 publishes the text encoder and transformer a second
    # time below ``vae/``. The duplicate paths resolve to the same immutable
    # Hub blobs as the primary components, so adding them costs no additional
    # model storage. They are nevertheless part of Diffusers' requested
    # safetensors closure, and huggingface_hub >= 1.28 correctly rejects an
    # offline snapshot when these aliases are absent.
    "vae/text_encoder/model-00001-of-00004.safetensors",
    "vae/text_encoder/model-00002-of-00004.safetensors",
    "vae/text_encoder/model-00003-of-00004.safetensors",
    "vae/text_encoder/model-00004-of-00004.safetensors",
    "vae/text_encoder/model.safetensors.index.json",
    "vae/transformer/diffusion_pytorch_model-00001-of-00006.safetensors",
    "vae/transformer/diffusion_pytorch_model-00002-of-00006.safetensors",
    "vae/transformer/diffusion_pytorch_model-00003-of-00006.safetensors",
    "vae/transformer/diffusion_pytorch_model-00004-of-00006.safetensors",
    "vae/transformer/diffusion_pytorch_model-00005-of-00006.safetensors",
    "vae/transformer/diffusion_pytorch_model-00006-of-00006.safetensors",
    "vae/transformer/diffusion_pytorch_model.safetensors.index.json",
]
LTX_VIDEO_FALLBACK_REPO = "Lightricks/LTX-Video"
ACE_STEP_REPO = "ACE-Step/acestep-v15-xl-turbo-diffusers"
ACE_STEP_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "condition_encoder/config.json",
    "condition_encoder/diffusion_pytorch_model.safetensors",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/added_tokens.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
ACE_STEP_LORA_BASE_REPO = "Runware/acestep-v15-turbo-diffusers"
STABLE_AUDIO_REPO = "stabilityai/stable-audio-open-1.0"
STABLE_AUDIO_DIFFUSERS_FILES = [
    ".gitattributes",
    "LICENSE.md",
    "README.md",
    "fma_dataset_attribution2.csv",
    "freesound_dataset_attribution2.csv",
    "model_index.json",
    "projection_model/config.json",
    "projection_model/diffusion_pytorch_model.safetensors",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
MINIMAX_MUSIC3_REPO = "MiniMaxAI/MiniMax-Music3"
MINIMAX_MUSIC3_DIFFUSERS_FILES = [
    "LICENSE",
    "condition_encoder/config.json",
    "condition_encoder/diffusion_pytorch_model.safetensors",
    "language_model/config.json",
    "language_model/generation_config.json",
    "language_model/model-00001-of-00004.safetensors",
    "language_model/model-00002-of-00004.safetensors",
    "language_model/model-00003-of-00004.safetensors",
    "language_model/model-00004-of-00004.safetensors",
    "language_model/model.safetensors.index.json",
    "modular_model_index.json",
    "rvq_depth_decoder/config.json",
    "rvq_depth_decoder/diffusion_pytorch_model.safetensors",
    "scheduler/scheduler_config.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vocoder/config.json",
    "vocoder/diffusion_pytorch_model.safetensors",
]
MINIMAX_H3_REPO = "MiniMaxAI/MiniMax-H3"
MINIMAX_H3_WORKFLOW_WEIGHT_BYTES = 144016405316
MINIMAX_H3_FULL_ROOT_WEIGHT_BYTES = 210296909532
ANIMA_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
ANIMA_DIFFUSERS_FILES = [
    "LICENSE.md",
    "modular_model_index.json",
    "scheduler/scheduler_config.json",
    "t5_tokenizer/tokenizer.json",
    "t5_tokenizer/tokenizer_config.json",
    "text_conditioner/config.json",
    "text_conditioner/diffusion_pytorch_model.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/chat_template.jinja",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
HUNYUAN_VIDEO_15_T2V_REPO = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
HUNYUAN_VIDEO_15_I2V_REPO = (
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled"
)
HUNYUAN_VIDEO_15_LICENSE_REPO = "tencent/HunyuanVideo-1.5"
_HUNYUAN_VIDEO_15_COMMON_FILES = [
    ".gitattributes",
    "README.md",
    "guider/guider_config.json",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    *[f"text_encoder/model-{index:05d}-of-00003.safetensors" for index in range(1, 4)],
    "text_encoder/model.safetensors.index.json",
    "text_encoder_2/config.json",
    "text_encoder_2/model.safetensors",
    "tokenizer/added_tokens.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/added_tokens.json",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
]
HUNYUAN_VIDEO_15_T2V_DIFFUSERS_FILES = [
    *_HUNYUAN_VIDEO_15_COMMON_FILES,
    "transformer/config.json",
    *[
        f"transformer/diffusion_pytorch_model-{index:05d}-of-00007.safetensors"
        for index in range(1, 8)
    ],
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model-00001-of-00002.safetensors",
    "vae/diffusion_pytorch_model-00002-of-00002.safetensors",
    "vae/diffusion_pytorch_model.safetensors.index.json",
]
HUNYUAN_VIDEO_15_I2V_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "feature_extractor/preprocessor_config.json",
    "image_encoder/config.json",
    "image_encoder/model.safetensors",
    *[path for path in _HUNYUAN_VIDEO_15_COMMON_FILES if path not in {".gitattributes", "README.md"}],
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
COSMOS3_NANO_REPO = "nvidia/Cosmos3-Nano"
COSMOS3_GUARDRAIL_REPO = "nvidia/Cosmos-Guardrail1"
COSMOS3_DISTILLED_T2I_REPO = "nvidia/Cosmos3-Super-Text2Image-4Step"
COSMOS3_DISTILLED_I2V_REPO = "nvidia/Cosmos3-Super-Image2Video-4Step"
COSMOS3_NANO_DIFFUSERS_FILES = [
    "README.md",
    "SAFETY.md",
    "modular_model_index.json",
    "scheduler/scheduler_config.json",
    "sound_tokenizer/config.json",
    "sound_tokenizer/diffusion_pytorch_model.safetensors",
    "text_tokenizer/added_tokens.json",
    "text_tokenizer/chat_template.jinja",
    "text_tokenizer/merges.txt",
    "text_tokenizer/special_tokens_map.json",
    "text_tokenizer/tokenizer.json",
    "text_tokenizer/tokenizer_config.json",
    "text_tokenizer/vocab.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00004-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00005-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00006-of-00007.safetensors",
    "transformer/diffusion_pytorch_model-00007-of-00007.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
_COSMOS3_DISTILLED_COMMON_DIFFUSERS_FILES = [
    "README.md",
    "SAFETY.md",
    "modular_model_index.json",
    "scheduler/scheduler_config.json",
    "text_tokenizer/added_tokens.json",
    "text_tokenizer/chat_template.jinja",
    "text_tokenizer/merges.txt",
    "text_tokenizer/special_tokens_map.json",
    "text_tokenizer/tokenizer.json",
    "text_tokenizer/tokenizer_config.json",
    "text_tokenizer/vocab.json",
    "transformer/config.json",
    *[
        f"transformer/diffusion_pytorch_model-{index:05d}-of-00027.safetensors"
        for index in range(1, 28)
    ],
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
COSMOS3_DISTILLED_T2I_DIFFUSERS_FILES = [
    *_COSMOS3_DISTILLED_COMMON_DIFFUSERS_FILES[:4],
    "sound_tokenizer/config.json",
    "sound_tokenizer/diffusion_pytorch_model.safetensors",
    *_COSMOS3_DISTILLED_COMMON_DIFFUSERS_FILES[4:],
]
COSMOS3_DISTILLED_I2V_DIFFUSERS_FILES = list(_COSMOS3_DISTILLED_COMMON_DIFFUSERS_FILES)
HELIOS_BASE_REPO = "BestWishYsh/Helios-Base"
HELIOS_PYRAMID_REPO = "BestWishYsh/Helios-Mid"
HELIOS_DISTILLED_REPO = "BestWishYsh/Helios-Distilled"
HELIOS_DIFFUSERS_FILES = [
    "guider/guider_config.json",
    "modular_model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00005.safetensors",
    "text_encoder/model-00002-of-00005.safetensors",
    "text_encoder/model-00003-of-00005.safetensors",
    "text_encoder/model-00004-of-00005.safetensors",
    "text_encoder/model-00005-of-00005.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00006.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00006.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00006.safetensors",
    "transformer/diffusion_pytorch_model-00004-of-00006.safetensors",
    "transformer/diffusion_pytorch_model-00005-of-00006.safetensors",
    "transformer/diffusion_pytorch_model-00006-of-00006.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
WAN_ANIMATE_2_REPO = "Wan-AI/Wan2.2-Animate-2-14B-Diffusers"
WAN_ANIMATE_2_DISTILLED_REPO = "Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers"
WAN_ANIMATE_2_DIFFUSERS_FILES = [
    "README.md",
    "configuration.json",
    "image_encoder/config.json",
    "image_encoder/model.safetensors",
    "image_processor/merges.txt",
    "image_processor/preprocessor_config.json",
    "image_processor/special_tokens_map.json",
    "image_processor/tokenizer.json",
    "image_processor/tokenizer_config.json",
    "image_processor/vocab.json",
    "model_index.json",
    "modular_model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00003.safetensors",
    "text_encoder/model-00002-of-00003.safetensors",
    "text_encoder/model-00003-of-00003.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00004.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00004.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00004.safetensors",
    "transformer/diffusion_pytorch_model-00004-of-00004.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
LONGCAT_AUDIO_DIT_REPO = "ruixiangma/LongCat-AudioDiT-1B-Diffusers"
LONGCAT_AUDIO_DIT_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
AUDIO_LDM2_REPO = "cvssp/audioldm2"
AUDIO_LDM2_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "feature_extractor/preprocessor_config.json",
    "language_model/config.json",
    "language_model/model.safetensors",
    "model_index.json",
    "projection_model/config.json",
    "projection_model/diffusion_pytorch_model.safetensors",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/spiece.model",
    "tokenizer_2/tokenizer.json",
    "tokenizer_2/tokenizer_config.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "vocoder/config.json",
    "vocoder/model.safetensors",
]
SHAP_E_REPO = "openai/shap-e"
SHAP_E_SAFE_COMPONENT_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "prior/config.json",
    "prior/diffusion_pytorch_model.fp16.safetensors",
    "renderer/config.json",
    "renderer/diffusion_pytorch_model.fp16.safetensors",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
]
SHAP_E_IMG2IMG_REPO = "openai/shap-e-img2img"
SHAP_E_IMG2IMG_SAFE_COMPONENT_FILES = [
    ".gitattributes",
    "README.md",
    "image_encoder/config.json",
    "image_encoder/model.fp16.safetensors",
    "image_processor/preprocessor_config.json",
    "model_index.json",
    "prior/config.json",
    "prior/diffusion_pytorch_model.fp16.safetensors",
    "renderer/config.json",
    "renderer/diffusion_pytorch_model.fp16.safetensors",
    "scheduler/scheduler_config.json",
]
WAN_22_T2V_A14B_REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
WAN_ANIMATE_REPO = "Wan-AI/Wan2.2-Animate-14B-Diffusers"
WAN_FLF_REPO = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"
LTX2_REPO = "Lightricks/LTX-2"
LTX2_DIFFUSERS_FILES = [
    ".gitattributes",
    "LICENSE",
    "README.md",
    "audio_vae/config.json",
    "audio_vae/diffusion_pytorch_model.safetensors",
    "connectors/config.json",
    "connectors/diffusion_pytorch_model.safetensors",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model.safetensors.index.json",
    *[f"text_encoder/model-{index:05d}-of-00011.safetensors" for index in range(1, 12)],
    "tokenizer/added_tokens.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/preprocessor_config.json",
    "tokenizer/processor_config.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer.model",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    *[
        f"transformer/diffusion_pytorch_model-{index:05d}-of-00008.safetensors"
        for index in range(1, 9)
    ],
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "vocoder/config.json",
    "vocoder/diffusion_pytorch_model.safetensors",
]
LTX2_IC_LORA_CANNY_REPO = "Lightricks/LTX-2-19b-IC-LoRA-Canny-Control"
LTX2_IC_LORA_CANNY_WEIGHT_NAME = "ltx-2-19b-ic-lora-canny-control.safetensors"
LTX2_IC_LORA_CANNY_FILES = [
    ".gitattributes",
    "README.md",
    LTX2_IC_LORA_CANNY_WEIGHT_NAME,
]
FRAMEPACK_REPO = "lllyasviel/FramePackI2V_HY"
FRAMEPACK_TRANSFORMER_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "config.json",
    "diffusion_pytorch_model-00001-of-00003.safetensors",
    "diffusion_pytorch_model-00002-of-00003.safetensors",
    "diffusion_pytorch_model-00003-of-00003.safetensors",
    "diffusion_pytorch_model.safetensors.index.json",
]
FRAMEPACK_BASE_REPO = "hunyuanvideo-community/HunyuanVideo"
FRAMEPACK_BASE_COMPONENT_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "text_encoder/model.safetensors.index.json",
    "text_encoder_2/config.json",
    "text_encoder_2/model.safetensors",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
FRAMEPACK_VISION_REPO = "lllyasviel/flux_redux_bfl"
FRAMEPACK_VISION_COMPONENT_FILES = [
    ".gitattributes",
    "README.md",
    "feature_extractor/preprocessor_config.json",
    "image_encoder/config.json",
    "image_encoder/model.safetensors",
]
STABLE_VIDEO_DIFFUSION_REPO = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
STABLE_VIDEO_DIFFUSION_FP16_FILES = [
    ".gitattributes",
    "LICENSE.md",
    "README.md",
    "feature_extractor/preprocessor_config.json",
    "image_encoder/config.json",
    "image_encoder/model.fp16.safetensors",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]
ANIMATEDIFF_MOTION_REPO = "guoyww/animatediff-motion-adapter-v1-5-2"
ANIMATEDIFF_MOTION_FP16_FILES = [
    ".gitattributes",
    "README.md",
    "config.json",
    "diffusion_pytorch_model.fp16.safetensors",
]
ANIMATELCM_MOTION_REPO = "wangfuyun/AnimateLCM"
ANIMATELCM_MOTION_FP16_FILES = [
    ".gitattributes",
    "README.md",
    "config.json",
    "diffusion_pytorch_model.fp16.safetensors",
    "AnimateLCM_sd15_t2v_lora.safetensors",
]
COGVIDEOX_2B_REPO = "zai-org/CogVideoX-2b"
COGVIDEOX_2B_DIFFUSERS_FILES = [
    ".gitattributes",
    "LICENSE",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/added_tokens.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
_VIDEO_T5_DIFFUSERS_FILES = [
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/added_tokens.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer_config.json",
]
ALLEGRO_REPO = "rhymes-ai/Allegro"
ALLEGRO_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    *_VIDEO_T5_DIFFUSERS_FILES,
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
LATTE_REPO = "maxin-cn/Latte-1"
LATTE_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    *_VIDEO_T5_DIFFUSERS_FILES,
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
MOCHI_REPO = "genmo/mochi-1-preview"
MOCHI_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    *_VIDEO_T5_DIFFUSERS_FILES,
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.bf16-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model.bf16-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model.bf16-00003-of-00003.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.bf16.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.bf16.safetensors",
]
SANA_VIDEO_REPO = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
SANA_VIDEO_DIFFUSERS_FILES = [
    ".gitattributes",
    "LICENSE",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer.model",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
QWEN_CONTROLNET_REPO = "InstantX/Qwen-Image-ControlNet-Union"
QWEN_CONTROLNET_FILES = [
    ".gitattributes",
    "README.md",
    "config.json",
    "diffusion_pytorch_model.safetensors",
]
QWEN_IMAGE_2512_REPO = "Qwen/Qwen-Image-2512"
QWEN_IMAGE_2512_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/added_tokens.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00009-of-00009.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
QWEN_IMAGE_EDIT_REPO = "Qwen/Qwen-Image-Edit"
QWEN_IMAGE_EDIT_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "processor/added_tokens.json",
    "processor/chat_template.jinja",
    "processor/merges.txt",
    "processor/preprocessor_config.json",
    "processor/special_tokens_map.json",
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "processor/video_preprocessor_config.json",
    "processor/vocab.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/added_tokens.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
    "transformer/diffusion_pytorch_model-00009-of-00009.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
QWEN_IMAGE_EDIT_PLUS_REPO = "Qwen/Qwen-Image-Edit-2511"
QWEN_IMAGE_EDIT_2511_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "processor/added_tokens.json",
    "processor/chat_template.jinja",
    "processor/merges.txt",
    "processor/preprocessor_config.json",
    "processor/special_tokens_map.json",
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "processor/video_preprocessor_config.json",
    "processor/vocab.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/added_tokens.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00005.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
    "transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
    "transformer/diffusion_pytorch_model-00005-of-00005.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
QWEN_IMAGE_LAYERED_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "processor/added_tokens.json",
    "processor/chat_template.json",
    "processor/merges.txt",
    "processor/preprocessor_config.json",
    "processor/special_tokens_map.json",
    "processor/tokenizer.json",
    "processor/tokenizer_config.json",
    "processor/vocab.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/added_tokens.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00005.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
    "transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
    "transformer/diffusion_pytorch_model-00005-of-00005.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
Z_IMAGE_REPO = "Tongyi-MAI/Z-Image-Turbo"
Z_IMAGE_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model-00001-of-00003.safetensors",
    "text_encoder/model-00002-of-00003.safetensors",
    "text_encoder/model-00003-of-00003.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/merges.txt",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
DDPM_CIFAR10_REPO = "google/ddpm-cifar10-32"
DDPM_CIFAR10_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "config.json",
    "diffusion_pytorch_model.safetensors",
    "model_index.json",
    "scheduler_config.json",
]
CONSISTENCY_IMAGENET64_REPO = "openai/diffusers-cd_imagenet64_l2"
CONSISTENCY_IMAGENET64_DIFFUSERS_FILES = [
    ".gitattributes",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
]

_COSMOS3_NANO_STRUCTURAL_MODES = (
    "text_to_image",
    "text_to_video",
    "image_to_video",
    "video_to_video",
    "text_to_video_with_audio",
    "image_to_video_with_audio",
    "video_to_video_with_audio",
)
_COSMOS3_DISTILLED_STRUCTURAL_MODES = ("text_to_image", "image_to_video")


def _cosmos3_guardrail_dependency(mode):
    return (
        {
            "id": "cosmos3-mandatory-safety-guardrail",
            "label": "NVIDIA Cosmos Guardrail 1",
            "repo": COSMOS3_GUARDRAIL_REPO,
            "revision": require_catalog_revision(COSMOS3_GUARDRAIL_REPO),
            "kind": "safety_checker",
            "requiredForModes": [mode],
            "description": (
                "Mandatory gated Cosmos text-and-video safety checker; access and the NVIDIA Open Model "
                "License must be acknowledged before any execution qualification."
            ),
        },
    )


_STUDIO_MODEL_DEPENDENCY_REQUIREMENTS = {
    **{
        ("Cosmos3OmniModularPipeline", mode): _cosmos3_guardrail_dependency(mode)
        for mode in _COSMOS3_NANO_STRUCTURAL_MODES
    },
    **{
        ("Cosmos3DistilledModularPipeline", mode): _cosmos3_guardrail_dependency(mode)
        for mode in _COSMOS3_DISTILLED_STRUCTURAL_MODES
    },
    ("LTX2ModularPipeline", "in_context_to_video"): (
        {
            "id": "ltx2-ic-lora-canny-control",
            "label": "LTX-2 19B IC-LoRA Canny Control",
            "repo": LTX2_IC_LORA_CANNY_REPO,
            "revision": require_catalog_revision(LTX2_IC_LORA_CANNY_REPO),
            "kind": "adapter",
            "requiredForModes": ["in_context_to_video"],
            "downloadFiles": LTX2_IC_LORA_CANNY_FILES,
            "description": (
                "Exact official Canny IC-LoRA used by the reviewed LTX-2 in-context reference-video workflow."
            ),
        },
    ),
    ("HunyuanVideoFramepackPipeline", "image_to_video"): (
        {
            "id": "framepack-hunyuan-base-components",
            "label": "FramePack HunyuanVideo base components",
            "repo": FRAMEPACK_BASE_REPO,
            "revision": require_catalog_revision(FRAMEPACK_BASE_REPO),
            "kind": "base",
            "requiredForModes": ["image_to_video"],
            "downloadFiles": FRAMEPACK_BASE_COMPONENT_FILES,
            "description": (
                "Exact safetensors scheduler, encoders, tokenizers, and VAE composed with the FramePack transformer."
            ),
        },
        {
            "id": "framepack-siglip-vision-components",
            "label": "FramePack SigLIP vision components",
            "repo": FRAMEPACK_VISION_REPO,
            "revision": require_catalog_revision(FRAMEPACK_VISION_REPO),
            "kind": "adapter",
            "requiredForModes": ["image_to_video"],
            "downloadFiles": FRAMEPACK_VISION_COMPONENT_FILES,
            "description": (
                "Exact SigLIP image processor and safetensors vision encoder used by FramePack conditioning."
            ),
        },
    ),
    ("AnimateDiffPipeline", "text_to_video"): (
        {
            "id": "animatediff-motion-adapter-v1-5-2",
            "label": "AnimateDiff SD1.5 v2 MotionAdapter",
            "repo": ANIMATEDIFF_MOTION_REPO,
            "revision": require_catalog_revision(ANIMATEDIFF_MOTION_REPO),
            "kind": "adapter",
            "requiredForModes": ["text_to_video"],
            "downloadFiles": ANIMATEDIFF_MOTION_FP16_FILES,
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
            "downloadFiles": ANIMATELCM_MOTION_FP16_FILES,
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
            "downloadFiles": SDXL_T2I_ADAPTER_CANNY_FP16_FILES,
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
            "downloadFiles": SDXL_CONTROLNET_CANNY_FP16_FILES,
            "description": "Required by the generic SDXL Canny ControlNet workflow.",
        },
    ),
    ("HunyuanDiTControlNetPipeline", "control_image"): (
        {
            "id": "hunyuan-dit-v1-2-controlnet-canny",
            "label": "Hunyuan-DiT v1.2 Canny ControlNet",
            "repo": HUNYUAN_DIT_CONTROLNET_CANNY_REPO,
            "revision": require_catalog_revision(HUNYUAN_DIT_CONTROLNET_CANNY_REPO),
            "kind": "controlnet",
            "requiredForModes": ["control_image"],
            "downloadFiles": HUNYUAN_DIT_CONTROLNET_CANNY_FILES,
            "description": "Exact safetensors Canny component for the reviewed Hunyuan-DiT v1.2 distilled recipe.",
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
            "downloadFiles": SD15_CONTROLNET_CANNY_FILES,
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
            "downloadFiles": QWEN_CONTROLNET_FILES,
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
            "requiredForModes": ["edit_image", "multi_image_reference_edit"],
            "description": "Redux supplies reference embeddings to the app-installed FLUX.1-dev base pipeline.",
        },
    ),
    ("FluxReduxPipeline", "multi_image_reference_edit"): (
        {
            "id": "flux-redux-base",
            "label": "FLUX.1-dev base pipeline",
            "repo": FLUX_DEV_REPO,
            "revision": require_catalog_revision(FLUX_DEV_REPO, model_type="FluxDevPipeline"),
            "kind": "base",
            "requiredForModes": ["edit_image", "multi_image_reference_edit"],
            "description": "Redux supplies reference embeddings to the app-installed FLUX.1-dev base pipeline.",
        },
    ),
}

_STUDIO_MODEL_DEPENDENCY_REQUIREMENTS[("QwenImageControlNetPipeline", "control_image")] = tuple(
    {
        **deepcopy(requirement),
        "description": "Required by the generic standard Diffusers Qwen control-image workflow.",
    }
    for requirement in _STUDIO_MODEL_DEPENDENCY_REQUIREMENTS[("QwenImageModularPipeline", "control_image")]
)

for _qwen_modular_control_mode in ("control_edit_image", "control_inpaint"):
    _STUDIO_MODEL_DEPENDENCY_REQUIREMENTS[("QwenImageModularPipeline", _qwen_modular_control_mode)] = tuple(
        {
            **deepcopy(requirement),
            "requiredForModes": [_qwen_modular_control_mode],
        }
        for requirement in _STUDIO_MODEL_DEPENDENCY_REQUIREMENTS[("QwenImageModularPipeline", "control_image")]
    )

for _sdxl_standard_control_mode in ("control_edit_image", "control_inpaint"):
    _STUDIO_MODEL_DEPENDENCY_REQUIREMENTS[("StableDiffusionXLControlNetPipeline", _sdxl_standard_control_mode)] = (
        tuple(
            {
                **deepcopy(requirement),
                "requiredForModes": [_sdxl_standard_control_mode],
            }
            for requirement in _STUDIO_MODEL_DEPENDENCY_REQUIREMENTS[
                ("StableDiffusionXLControlNetPipeline", "control_image")
            ]
        )
    )

for _sdxl_modular_control_mode in ("control_image", "control_edit_image", "control_inpaint"):
    _STUDIO_MODEL_DEPENDENCY_REQUIREMENTS[("StableDiffusionXLModularPipeline", _sdxl_modular_control_mode)] = tuple(
        {
            **deepcopy(requirement),
            "requiredForModes": [_sdxl_modular_control_mode],
            "description": "Required by the pinned Modular Diffusers SDXL Canny ControlNet workflow.",
        }
        for requirement in _STUDIO_MODEL_DEPENDENCY_REQUIREMENTS[
            ("StableDiffusionXLControlNetPipeline", _sdxl_modular_control_mode)
        ]
    )

_SDXL_UNION_MODES = (
    "control_union_image",
    "control_union_edit_image",
    "control_union_inpaint",
)
_SDXL_IP_ADAPTER_MODES = (
    "ip_adapter_image",
    "ip_adapter_edit_image",
    "ip_adapter_inpaint",
)
_SDXL_IP_ADAPTER_CONTROL_MODES = (
    "ip_adapter_control_image",
    "ip_adapter_control_edit_image",
    "ip_adapter_control_inpaint",
)
_SDXL_IP_ADAPTER_UNION_MODES = (
    "ip_adapter_control_union_image",
    "ip_adapter_control_union_edit_image",
    "ip_adapter_control_union_inpaint",
)


def _sdxl_auxiliary_requirement(kind: str, mode: str) -> dict[str, Any]:
    if kind == "controlnet_union":
        return {
            "id": "sdxl-controlnet-union",
            "label": "Stable Diffusion XL ControlNet Union",
            "repo": SDXL_CONTROLNET_UNION_REPO,
            "revision": require_catalog_revision(SDXL_CONTROLNET_UNION_REPO),
            "kind": "controlnet",
            "requiredForModes": [mode],
            "downloadFiles": SDXL_CONTROLNET_UNION_FILES,
            "description": "Exact basic SDXL ControlNet Union artifact loaded through ControlNetUnionModel.",
        }
    if kind == "ip_adapter":
        return {
            "id": "sdxl-ip-adapter",
            "label": "Stable Diffusion XL IP-Adapter",
            "repo": SDXL_IP_ADAPTER_REPO,
            "revision": require_catalog_revision(SDXL_IP_ADAPTER_REPO),
            "kind": "adapter",
            "requiredForModes": [mode],
            "downloadFiles": SDXL_IP_ADAPTER_FILES,
            "description": "Exact SDXL IP-Adapter weight and CLIP vision encoder used by the modular block.",
        }
    raise ValueError(f"Unknown SDXL auxiliary requirement {kind!r}.")


for _mode in _SDXL_UNION_MODES:
    _STUDIO_MODEL_DEPENDENCY_REQUIREMENTS[("StableDiffusionXLModularPipeline", _mode)] = (
        _sdxl_auxiliary_requirement("controlnet_union", _mode),
    )
for _mode in _SDXL_IP_ADAPTER_MODES:
    _STUDIO_MODEL_DEPENDENCY_REQUIREMENTS[("StableDiffusionXLModularPipeline", _mode)] = (
        _sdxl_auxiliary_requirement("ip_adapter", _mode),
    )
for _mode in _SDXL_IP_ADAPTER_CONTROL_MODES:
    _ordinary_mode = _mode.removeprefix("ip_adapter_")
    _ordinary = deepcopy(
        _STUDIO_MODEL_DEPENDENCY_REQUIREMENTS[("StableDiffusionXLModularPipeline", _ordinary_mode)][0]
    )
    _ordinary["requiredForModes"] = [_mode]
    _STUDIO_MODEL_DEPENDENCY_REQUIREMENTS[("StableDiffusionXLModularPipeline", _mode)] = (
        _ordinary,
        _sdxl_auxiliary_requirement("ip_adapter", _mode),
    )
for _mode in _SDXL_IP_ADAPTER_UNION_MODES:
    _STUDIO_MODEL_DEPENDENCY_REQUIREMENTS[("StableDiffusionXLModularPipeline", _mode)] = (
        _sdxl_auxiliary_requirement("controlnet_union", _mode),
        _sdxl_auxiliary_requirement("ip_adapter", _mode),
    )

for _animatediff_model_type, _animatediff_mode, _uses_controlnet in (
    ("AnimateDiffPAGPipeline", "text_to_video", False),
    ("AnimateDiffVideoToVideoPipeline", "video_to_video", False),
    ("AnimateDiffControlNetPipeline", "control_to_video", True),
    (
        "AnimateDiffVideoToVideoControlNetPipeline",
        "control_video_to_video",
        True,
    ),
):
    _animatediff_requirements = [
        {
            **deepcopy(_requirement),
            "requiredForModes": [_animatediff_mode],
            "description": (
                f"Required by the generic {_animatediff_model_type} {_animatediff_mode.replace('_', ' ')} workflow."
            ),
        }
        for _requirement in _STUDIO_MODEL_DEPENDENCY_REQUIREMENTS[("AnimateDiffPipeline", "text_to_video")]
    ]
    if _uses_controlnet:
        _animatediff_requirements.extend(
            {
                **deepcopy(_requirement),
                "requiredForModes": [_animatediff_mode],
                "description": (
                    f"Required by the generic {_animatediff_model_type} "
                    f"{_animatediff_mode.replace('_', ' ')} workflow."
                ),
            }
            for _requirement in _STUDIO_MODEL_DEPENDENCY_REQUIREMENTS[("StableDiffusionPipeline", "control_image")]
        )
    _STUDIO_MODEL_DEPENDENCY_REQUIREMENTS[(_animatediff_model_type, _animatediff_mode)] = tuple(
        _animatediff_requirements
    )

for _control_model_type, _source_model_type, _control_modes in (
    (
        "StableDiffusionPipeline",
        "StableDiffusionPipeline",
        ("control_edit_image", "control_inpaint"),
    ),
    (
        "StableDiffusionPAGPipeline",
        "StableDiffusionPipeline",
        ("control_image", "control_inpaint"),
    ),
    (
        "StableDiffusionXLControlNetPipeline",
        "StableDiffusionXLControlNetPipeline",
        ("control_edit_image", "control_inpaint"),
    ),
    (
        "StableDiffusionXLPAGPipeline",
        "StableDiffusionXLControlNetPipeline",
        ("control_image", "control_edit_image"),
    ),
):
    for _control_mode in _control_modes:
        _STUDIO_MODEL_DEPENDENCY_REQUIREMENTS[(_control_model_type, _control_mode)] = tuple(
            {
                **deepcopy(_requirement),
                "requiredForModes": [_control_mode],
                "description": (
                    f"Required by the generic {_control_model_type} {_control_mode.replace('_', ' ')} workflow."
                ),
            }
            for _requirement in _STUDIO_MODEL_DEPENDENCY_REQUIREMENTS[(_source_model_type, "control_image")]
        )


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

# Wan 2.2 A14B contains two roughly 14B denoising experts. Keeping the full
# BF16 pipeline resident in host memory while a component is placed on a
# shared-memory ROCm accelerator exceeded a 121 GiB host during the reviewed
# 2026-08-29 frontend qualification run (about 112 GiB process RSS before the
# kernel OOM kill). Disk-group offload removes inactive groups from host RAM,
# but needs enough free disk for the additional safetensors plus a useful
# post-offload reserve. These are execution-admission limits, not model-card
# quality recommendations.
_WAN_22_A14B_EXPERT_RESOURCE_REQUIREMENTS = {
    OFFLOAD_MODE_NONE: {
        "accelerator": "cuda",
        "vramBytes": 80 * _GIB,
        "systemRamBytes": 96 * _GIB,
        "diskFreeBytes": 140 * _GIB,
    },
    OFFLOAD_MODE_MODEL_CPU: {
        "accelerator": "cuda",
        "vramBytes": 24 * _GIB,
        "systemRamBytes": 160 * _GIB,
        "diskFreeBytes": 140 * _GIB,
    },
    OFFLOAD_MODE_SEQUENTIAL_CPU: {
        "accelerator": "cuda",
        "vramBytes": 24 * _GIB,
        "systemRamBytes": 160 * _GIB,
        "diskFreeBytes": 140 * _GIB,
    },
    OFFLOAD_MODE_GROUP_CPU: {
        "accelerator": "cuda",
        "vramBytes": 24 * _GIB,
        "systemRamBytes": 160 * _GIB,
        "diskFreeBytes": 140 * _GIB,
    },
    OFFLOAD_MODE_GROUP_DISK: {
        "accelerator": "cuda",
        "vramBytes": 24 * _GIB,
        "systemRamBytes": 96 * _GIB,
        "diskFreeBytes": 140 * _GIB,
    },
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
_SDXL_GRAPH_BINDINGS = _GRAPH_BINDINGS + (("diffusersImagePipeline", "revision", "defaultRevision"),)
_PAG_GRAPH_BINDINGS = _SDXL_GRAPH_BINDINGS + (
    ("diffusersImageGenerate", "pag_scale", "pagScale"),
    ("diffusersImageGenerate", "pag_adaptive_scale", "pagAdaptiveScale"),
)
_PAG_TEXT_TO_IMAGE_BINDINGS = tuple(
    item for item in _PAG_GRAPH_BINDINGS if item[:2] != ("diffusersImageGenerate", "strength")
)
_HUNYUAN_PAG_GRAPH_BINDINGS = tuple(
    item for item in _PAG_TEXT_TO_IMAGE_BINDINGS if item[:2] != ("diffusersImageGenerate", "max_sequence_length")
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
    ("speechModel", "pipeline_class", "pipelineClass"),
    ("speechModel", "execution_profile_id", "executionProfileId"),
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
_CTC_SPEECH_GRAPH_ROLES = (
    ("speechModel", "modules.HuggingFaceSpeech.LoadCTCSpeechRecognitionModel", -720, -80),
    ("loadAudio", "modules.Audio.Load", -720, 280),
    ("transcribeAudio", "modules.HuggingFaceSpeech.TranscribeCTCAudio", -240, -80),
    ("transcriptPreview", "modules.Primitive.DataViewer", 240, -80),
)
_CTC_SPEECH_GRAPH_EDGES = _SPEECH_GRAPH_EDGES
_CTC_SPEECH_GRAPH_BINDINGS = (
    ("speechModel", "model_id", "artifact"),
    ("speechModel", "pipeline_class", "pipelineClass"),
    ("speechModel", "execution_profile_id", "executionProfileId"),
    ("speechModel", "revision", "defaultRevision"),
    ("speechModel", "dtype", "dtype"),
    ("speechModel", "device", "device"),
    ("loadAudio", "file", "sourceAudio"),
    ("transcribeAudio", "timestamps", "speechTimestamps"),
    ("transcribeAudio", "chunk_length_seconds", "speechChunkSeconds"),
    ("transcribeAudio", "stride_length_seconds", "speechStrideSeconds"),
)
_TRANSFORMERS_TEXT_GRAPH_ROLES = (
    (
        "transformersTextModel",
        "modules.HuggingFaceTransformers.LoadTextGenerationModel",
        -720,
        -80,
    ),
    ("transformersTextGenerate", "modules.HuggingFaceTransformers.GenerateText", -240, -80),
    ("transformersTextPreview", "modules.Primitive.DataViewer", 240, -80),
)
_TRANSFORMERS_TEXT_GRAPH_EDGES = (
    ("transformersTextModel", "model", "transformersTextGenerate", "model"),
    ("transformersTextGenerate", "result", "transformersTextPreview", "value"),
)
_TRANSFORMERS_TEXT_GRAPH_BINDINGS = (
    ("transformersTextModel", "model_id", "artifact"),
    ("transformersTextModel", "revision", "defaultRevision"),
    ("transformersTextModel", "dtype", "dtype"),
    ("transformersTextModel", "device", "device"),
    ("transformersTextModel", "quantization_mode", "quantizationMode"),
    ("transformersTextGenerate", "prompt", "prompt"),
    ("transformersTextGenerate", "use_chat_template", "useChatTemplate"),
    ("transformersTextGenerate", "max_new_tokens", "maxNewTokens"),
    ("transformersTextGenerate", "min_new_tokens", "minNewTokens"),
    ("transformersTextGenerate", "do_sample", "doSample"),
    ("transformersTextGenerate", "temperature", "temperature"),
    ("transformersTextGenerate", "top_p", "topP"),
    ("transformersTextGenerate", "top_k", "topK"),
    ("transformersTextGenerate", "num_beams", "numBeams"),
    ("transformersTextGenerate", "repetition_penalty", "repetitionPenalty"),
)
_TRANSFORMERS_IMAGE_TEXT_GRAPH_ROLES = (
    (
        "transformersImageTextModel",
        "modules.HuggingFaceTransformers.LoadImageTextToTextModel",
        -720,
        -80,
    ),
    ("loadImage", "modules.Image.Load", -720, 280),
    (
        "transformersImageTextGenerate",
        "modules.HuggingFaceTransformers.GenerateImageVideoText",
        -240,
        -80,
    ),
    ("transformersTextPreview", "modules.Primitive.DataViewer", 240, -80),
)
_TRANSFORMERS_IMAGE_TEXT_GRAPH_EDGES = (
    ("transformersImageTextModel", "model", "transformersImageTextGenerate", "model"),
    ("loadImage", "image", "transformersImageTextGenerate", "images"),
    ("transformersImageTextGenerate", "result", "transformersTextPreview", "value"),
)
_TRANSFORMERS_IMAGE_TEXT_GRAPH_BINDINGS = (
    ("transformersImageTextModel", "model_id", "artifact"),
    ("transformersImageTextModel", "revision", "defaultRevision"),
    ("transformersImageTextModel", "dtype", "dtype"),
    ("transformersImageTextModel", "device", "device"),
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "alphaMode"),
    ("transformersImageTextGenerate", "prompt", "prompt"),
    ("transformersImageTextGenerate", "use_chat_template", "useChatTemplate"),
    ("transformersImageTextGenerate", "max_new_tokens", "maxNewTokens"),
    ("transformersImageTextGenerate", "min_new_tokens", "minNewTokens"),
    ("transformersImageTextGenerate", "do_sample", "doSample"),
    ("transformersImageTextGenerate", "temperature", "temperature"),
    ("transformersImageTextGenerate", "top_p", "topP"),
    ("transformersImageTextGenerate", "top_k", "topK"),
    ("transformersImageTextGenerate", "num_beams", "numBeams"),
    ("transformersImageTextGenerate", "repetition_penalty", "repetitionPenalty"),
)
_TRANSFORMERS_ANY_TO_ANY_TEXT_GRAPH_ROLES = (
    (
        "transformersAnyToAnyModel",
        "modules.HuggingFaceTransformers.LoadAnyToAnyModel",
        -720,
        -80,
    ),
    (
        "transformersAnyToAnyGenerate",
        "modules.HuggingFaceTransformers.GenerateAnyToAny",
        -240,
        -80,
    ),
    ("transformersTextPreview", "modules.Primitive.DataViewer", 240, -80),
)
_TRANSFORMERS_ANY_TO_ANY_TEXT_GRAPH_EDGES = (
    ("transformersAnyToAnyModel", "model", "transformersAnyToAnyGenerate", "model"),
    ("transformersAnyToAnyGenerate", "result", "transformersTextPreview", "value"),
)
_TRANSFORMERS_ANY_TO_ANY_TEXT_GRAPH_BINDINGS = (
    ("transformersAnyToAnyModel", "model_id", "artifact"),
    ("transformersAnyToAnyModel", "revision", "defaultRevision"),
    ("transformersAnyToAnyModel", "dtype", "dtype"),
    ("transformersAnyToAnyModel", "device", "device"),
    ("transformersAnyToAnyGenerate", "prompt", "prompt"),
    ("transformersAnyToAnyGenerate", "generation_mode", "anyToAnyText"),
    ("transformersAnyToAnyGenerate", "max_new_tokens", "maxNewTokens"),
    ("transformersAnyToAnyGenerate", "min_new_tokens", "minNewTokens"),
    ("transformersAnyToAnyGenerate", "do_sample", "doSample"),
    ("transformersAnyToAnyGenerate", "temperature", "temperature"),
    ("transformersAnyToAnyGenerate", "top_p", "topP"),
    ("transformersAnyToAnyGenerate", "top_k", "topK"),
    ("transformersAnyToAnyGenerate", "num_beams", "numBeams"),
    ("transformersAnyToAnyGenerate", "repetition_penalty", "repetitionPenalty"),
)
_TRANSFORMERS_ANY_TO_ANY_IMAGE_TEXT_GRAPH_ROLES = (
    (
        "transformersAnyToAnyModel",
        "modules.HuggingFaceTransformers.LoadAnyToAnyModel",
        -720,
        -80,
    ),
    ("loadImage", "modules.Image.Load", -720, 280),
    (
        "transformersAnyToAnyGenerate",
        "modules.HuggingFaceTransformers.GenerateAnyToAny",
        -240,
        -80,
    ),
    ("transformersTextPreview", "modules.Primitive.DataViewer", 240, -80),
)
_TRANSFORMERS_ANY_TO_ANY_IMAGE_TEXT_GRAPH_EDGES = (
    ("transformersAnyToAnyModel", "model", "transformersAnyToAnyGenerate", "model"),
    ("loadImage", "image", "transformersAnyToAnyGenerate", "images"),
    ("transformersAnyToAnyGenerate", "result", "transformersTextPreview", "value"),
)
_TRANSFORMERS_ANY_TO_ANY_IMAGE_TEXT_GRAPH_BINDINGS = (
    ("transformersAnyToAnyModel", "model_id", "artifact"),
    ("transformersAnyToAnyModel", "revision", "defaultRevision"),
    ("transformersAnyToAnyModel", "dtype", "dtype"),
    ("transformersAnyToAnyModel", "device", "device"),
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "alphaMode"),
    ("transformersAnyToAnyGenerate", "prompt", "prompt"),
    ("transformersAnyToAnyGenerate", "generation_mode", "anyToAnyText"),
    ("transformersAnyToAnyGenerate", "max_new_tokens", "maxNewTokens"),
    ("transformersAnyToAnyGenerate", "min_new_tokens", "minNewTokens"),
    ("transformersAnyToAnyGenerate", "do_sample", "doSample"),
    ("transformersAnyToAnyGenerate", "temperature", "temperature"),
    ("transformersAnyToAnyGenerate", "top_p", "topP"),
    ("transformersAnyToAnyGenerate", "top_k", "topK"),
    ("transformersAnyToAnyGenerate", "num_beams", "numBeams"),
    ("transformersAnyToAnyGenerate", "repetition_penalty", "repetitionPenalty"),
)
_TRANSFORMERS_ANY_TO_ANY_IMAGE_GRAPH_ROLES = (
    (
        "transformersAnyToAnyModel",
        "modules.HuggingFaceTransformers.LoadAnyToAnyModel",
        -720,
        -80,
    ),
    (
        "transformersAnyToAnyGenerate",
        "modules.HuggingFaceTransformers.GenerateAnyToAny",
        -240,
        -80,
    ),
    ("preview", "modules.Image.Preview", 240, -80),
)
_TRANSFORMERS_ANY_TO_ANY_IMAGE_GRAPH_EDGES = (
    ("transformersAnyToAnyModel", "model", "transformersAnyToAnyGenerate", "model"),
    ("transformersAnyToAnyGenerate", "image", "preview", "image"),
)
_TRANSFORMERS_ANY_TO_ANY_IMAGE_GRAPH_BINDINGS = (
    ("transformersAnyToAnyModel", "model_id", "artifact"),
    ("transformersAnyToAnyModel", "revision", "defaultRevision"),
    ("transformersAnyToAnyModel", "dtype", "dtype"),
    ("transformersAnyToAnyModel", "device", "device"),
    ("transformersAnyToAnyGenerate", "prompt", "prompt"),
    ("transformersAnyToAnyGenerate", "generation_mode", "anyToAnyImage"),
    # Janus image tokens are sampled autoregressively. The official
    # Transformers recipe requires do_sample=True; leaving the generic node's
    # text-oriented default in place silently turns this into greedy decoding.
    ("transformersAnyToAnyGenerate", "do_sample", "true"),
    ("transformersAnyToAnyGenerate", "max_new_tokens", "maxNewTokens"),
    ("transformersAnyToAnyGenerate", "min_new_tokens", "minNewTokens"),
    ("transformersAnyToAnyGenerate", "temperature", "temperature"),
    ("transformersAnyToAnyGenerate", "top_p", "topP"),
    ("transformersAnyToAnyGenerate", "top_k", "topK"),
    ("transformersAnyToAnyGenerate", "num_beams", "numBeams"),
    ("transformersAnyToAnyGenerate", "repetition_penalty", "repetitionPenalty"),
)
_MODULAR_TEXT_TO_IMAGE_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -720, -80),
    ("prompt", "modules.ModularDiffusers.EncodePrompt", -360, -240),
    ("denoise", "modules.ModularDiffusers.Denoise", 80, -80),
    ("decode", "modules.ModularDiffusers.DecodeLatents", 440, -80),
    ("preview", "modules.Image.Preview", 800, -80),
)
_MODULAR_TEXT_TO_IMAGE_GRAPH_EDGES = (
    ("models", "text_encoders", "prompt", "text_encoders"),
    ("models", "unet_out", "denoise", "unet"),
    ("models", "scheduler", "denoise", "scheduler"),
    ("models", "vae_out", "denoise", "vae"),
    ("models", "vae_out", "decode", "vae"),
    ("prompt", "embeddings", "denoise", "embeddings"),
    ("denoise", "latents", "decode", "latents"),
    ("denoise", "route_state_out", "decode", "route_state_in"),
    ("decode", "images", "preview", "image"),
)
_MODULAR_TEXT_TO_IMAGE_GRAPH_BINDINGS = (
    ("models", "model_type", "pipelineClass"),
    ("models", "repo_id", "artifact"),
    ("models", "revision", "defaultRevision"),
    ("models", "dtype", "dtype"),
    ("models", "device", "device"),
    ("models", "auto_offload", "autoOffload"),
    ("models", "offload_mode", "offloadMode"),
    ("models", "trust_remote_code", "false"),
    ("prompt", "prompt", "prompt"),
    ("prompt", "negative_prompt", "negativePrompt"),
    ("denoise", "width", "width"),
    ("denoise", "height", "height"),
    ("denoise", "seed", "seed"),
    ("denoise", "num_inference_steps", "steps"),
    ("denoise", "guidance_scale", "guidanceScale"),
)
_MODULAR_FLUX_TEXT_TO_IMAGE_GRAPH_EDGES = tuple(
    edge
    for edge in _MODULAR_TEXT_TO_IMAGE_GRAPH_EDGES
    if edge
    not in {
        ("models", "vae_out", "denoise", "vae"),
        ("denoise", "route_state_out", "decode", "route_state_in"),
    }
)
_MODULAR_FLUX_TEXT_TO_IMAGE_GRAPH_BINDINGS = tuple(
    binding
    for binding in _MODULAR_TEXT_TO_IMAGE_GRAPH_BINDINGS
    if binding != ("prompt", "negative_prompt", "negativePrompt")
)
_MODULAR_FLUX_DECODE_GEOMETRY_EDGES = (
    ("denoise", "out_width", "decode", "width"),
    ("denoise", "out_height", "decode", "height"),
)
_MODULAR_FLUX_IMAGE_ENCODE_GEOMETRY_BINDINGS = (
    ("imageEncode", "width", "optionalWidth"),
    ("imageEncode", "height", "optionalHeight"),
)
_MODULAR_SDXL_EDIT_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -720, -80),
    ("prompt", "modules.ModularDiffusers.EncodePrompt", -360, -240),
    ("loadImage", "modules.Image.Load", -720, 320),
    ("imageEncode", "modules.ModularDiffusers.ImageEncode", -360, 320),
    ("denoise", "modules.ModularDiffusers.Denoise", 80, -80),
    ("decode", "modules.ModularDiffusers.DecodeLatents", 440, -80),
    ("preview", "modules.Image.Preview", 800, -80),
)
_MODULAR_SDXL_EDIT_GRAPH_EDGES = (
    ("models", "text_encoders", "prompt", "text_encoders"),
    ("models", "unet_out", "denoise", "unet"),
    ("models", "scheduler", "denoise", "scheduler"),
    ("models", "vae_out", "imageEncode", "vae"),
    ("models", "vae_out", "denoise", "vae"),
    ("models", "vae_out", "decode", "vae"),
    ("loadImage", "image", "imageEncode", "image"),
    ("prompt", "embeddings", "denoise", "embeddings"),
    ("imageEncode", "image_latents", "denoise", "image_latents"),
    ("imageEncode", "route_state_out", "denoise", "route_state_in"),
    ("denoise", "latents", "decode", "latents"),
    ("denoise", "route_state_out", "decode", "route_state_in"),
    ("decode", "images", "preview", "image"),
)
_MODULAR_SDXL_EDIT_GRAPH_BINDINGS = (
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
    ("prompt", "prompt", "prompt"),
    ("prompt", "negative_prompt", "negativePrompt"),
    ("imageEncode", "seed", "seed"),
    ("denoise", "width", "width"),
    ("denoise", "height", "height"),
    ("denoise", "seed", "seed"),
    ("denoise", "num_inference_steps", "steps"),
    ("denoise", "guidance_scale", "guidanceScale"),
    ("denoise", "strength", "strength"),
)
_MODULAR_FLUX_IMAGE_TO_IMAGE_GRAPH_EDGES = tuple(
    edge
    for edge in _MODULAR_SDXL_EDIT_GRAPH_EDGES
    if edge
    not in {
        ("models", "vae_out", "denoise", "vae"),
        ("imageEncode", "route_state_out", "denoise", "route_state_in"),
        ("denoise", "route_state_out", "decode", "route_state_in"),
    }
)
_MODULAR_Z_IMAGE_TEXT_TO_IMAGE_GRAPH_EDGES = _MODULAR_FLUX_TEXT_TO_IMAGE_GRAPH_EDGES
_MODULAR_Z_IMAGE_TO_IMAGE_GRAPH_EDGES = _MODULAR_FLUX_IMAGE_TO_IMAGE_GRAPH_EDGES
_MODULAR_SDXL_INPAINT_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -720, -80),
    ("prompt", "modules.ModularDiffusers.EncodePrompt", -360, -240),
    ("loadImage", "modules.Image.Load", -720, 280),
    ("loadMask", "modules.Image.Load", -720, 560),
    ("imageEncode", "modules.ModularDiffusers.ImageEncode", -360, 360),
    ("denoise", "modules.ModularDiffusers.Denoise", 80, -80),
    ("decode", "modules.ModularDiffusers.DecodeLatents", 440, -80),
    ("preview", "modules.Image.Preview", 800, -80),
)
_MODULAR_SDXL_INPAINT_GRAPH_EDGES = (
    ("models", "text_encoders", "prompt", "text_encoders"),
    ("models", "unet_out", "denoise", "unet"),
    ("models", "scheduler", "denoise", "scheduler"),
    ("models", "vae_out", "imageEncode", "vae"),
    ("models", "vae_out", "denoise", "vae"),
    ("models", "vae_out", "decode", "vae"),
    ("loadImage", "image", "imageEncode", "image"),
    ("loadMask", "image", "imageEncode", "mask_image"),
    ("prompt", "embeddings", "denoise", "embeddings"),
    ("imageEncode", "image_latents", "denoise", "image_latents"),
    ("imageEncode", "mask", "denoise", "mask"),
    ("imageEncode", "masked_image_latents", "denoise", "masked_image_latents"),
    ("imageEncode", "route_state_out", "denoise", "route_state_in"),
    ("denoise", "latents", "decode", "latents"),
    ("denoise", "route_state_out", "decode", "route_state_in"),
    ("decode", "images", "preview", "image"),
)
_MODULAR_SDXL_INPAINT_GRAPH_BINDINGS = (
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
    ("loadMask", "file", "maskImage"),
    ("loadMask", "alpha_channel", "removeAlpha"),
    ("prompt", "prompt", "prompt"),
    ("prompt", "negative_prompt", "negativePrompt"),
    ("imageEncode", "seed", "seed"),
    ("denoise", "width", "width"),
    ("denoise", "height", "height"),
    ("denoise", "seed", "seed"),
    ("denoise", "num_inference_steps", "steps"),
    ("denoise", "guidance_scale", "guidanceScale"),
    ("denoise", "strength", "strength"),
)
_MODULAR_SDXL_CONTROL_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -720, -80),
    ("prompt", "modules.ModularDiffusers.EncodePrompt", -360, -240),
    ("loadImage", "modules.Image.Load", -720, 320),
    ("controlnetModel", "modules.ModularDiffusers.AutoModelLoader", -360, 520),
    ("controlnet", "modules.ModularDiffusers.Controlnet", 80, 320),
    ("denoise", "modules.ModularDiffusers.Denoise", 80, -80),
    ("decode", "modules.ModularDiffusers.DecodeLatents", 440, -80),
    ("preview", "modules.Image.Preview", 800, -80),
)
_MODULAR_SDXL_CONTROL_GRAPH_EDGES = (
    ("controlnetModel", "model", "models", "controlnet"),
    ("models", "text_encoders", "prompt", "text_encoders"),
    ("models", "unet_out", "denoise", "unet"),
    ("models", "scheduler", "denoise", "scheduler"),
    ("models", "vae_out", "denoise", "vae"),
    ("models", "vae_out", "decode", "vae"),
    ("loadImage", "image", "controlnet", "control_image"),
    ("controlnetModel", "model", "controlnet", "controlnet"),
    ("prompt", "embeddings", "denoise", "embeddings"),
    ("controlnet", "controlnet_bundle", "denoise", "controlnet_bundle"),
    ("denoise", "latents", "decode", "latents"),
    ("denoise", "route_state_out", "decode", "route_state_in"),
    ("decode", "images", "preview", "image"),
)
_MODULAR_SDXL_CONTROL_GRAPH_BINDINGS = (
    ("models", "model_type", "pipelineClass"),
    ("models", "repo_id", "artifact"),
    ("models", "revision", "defaultRevision"),
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
    ("controlnetModel", "variant", "fp16"),
    ("controlnetModel", "trust_remote_code", "false"),
    ("controlnetModel", "revision", "revision"),
    ("controlnetModel", "device", "device"),
    ("controlnetModel", "auto_offload", "autoOffload"),
    ("controlnetModel", "offload_mode", "offloadMode"),
    ("prompt", "prompt", "prompt"),
    ("prompt", "negative_prompt", "negativePrompt"),
    ("controlnet", "model_type", "pipelineClass"),
    ("controlnet", "controlnet_variant", "ordinary"),
    ("controlnet", "width", "width"),
    ("controlnet", "height", "height"),
    ("controlnet", "controlnet_conditioning_scale", "conditioningScale"),
    ("denoise", "width", "width"),
    ("denoise", "height", "height"),
    ("denoise", "seed", "seed"),
    ("denoise", "num_inference_steps", "steps"),
    ("denoise", "guidance_scale", "guidanceScale"),
)
_MODULAR_SDXL_CONTROL_EDIT_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -880, -80),
    ("prompt", "modules.ModularDiffusers.EncodePrompt", -520, -240),
    ("loadImage", "modules.Image.Load", -880, 240),
    ("imageEncode", "modules.ModularDiffusers.ImageEncode", -520, 240),
    ("loadControlImage", "modules.Image.Load", -880, 520),
    ("controlnetModel", "modules.ModularDiffusers.AutoModelLoader", -520, 640),
    ("controlnet", "modules.ModularDiffusers.Controlnet", -160, 440),
    ("denoise", "modules.ModularDiffusers.Denoise", 160, -80),
    ("decode", "modules.ModularDiffusers.DecodeLatents", 520, -80),
    ("preview", "modules.Image.Preview", 880, -80),
)
_MODULAR_SDXL_CONTROL_EDIT_GRAPH_EDGES = (
    ("controlnetModel", "model", "models", "controlnet"),
    ("models", "text_encoders", "prompt", "text_encoders"),
    ("models", "unet_out", "denoise", "unet"),
    ("models", "scheduler", "denoise", "scheduler"),
    ("models", "vae_out", "imageEncode", "vae"),
    ("models", "vae_out", "denoise", "vae"),
    ("models", "vae_out", "decode", "vae"),
    ("loadImage", "image", "imageEncode", "image"),
    ("loadControlImage", "image", "controlnet", "control_image"),
    ("controlnetModel", "model", "controlnet", "controlnet"),
    ("prompt", "embeddings", "denoise", "embeddings"),
    ("imageEncode", "image_latents", "denoise", "image_latents"),
    ("imageEncode", "route_state_out", "denoise", "route_state_in"),
    ("controlnet", "controlnet_bundle", "denoise", "controlnet_bundle"),
    ("denoise", "latents", "decode", "latents"),
    ("denoise", "route_state_out", "decode", "route_state_in"),
    ("decode", "images", "preview", "image"),
)
_MODULAR_SDXL_CONTROL_EDIT_GRAPH_BINDINGS = (
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
    ("imageEncode", "seed", "seed"),
    ("loadControlImage", "file", "controlImage"),
    ("loadControlImage", "alpha_channel", "alphaMode"),
    ("controlnetModel", "model_type", "kind"),
    ("controlnetModel", "model_id", "repo"),
    ("controlnetModel", "dtype", "dtype"),
    ("controlnetModel", "subfolder", "empty"),
    ("controlnetModel", "variant", "fp16"),
    ("controlnetModel", "trust_remote_code", "false"),
    ("controlnetModel", "revision", "revision"),
    ("controlnetModel", "device", "device"),
    ("controlnetModel", "auto_offload", "autoOffload"),
    ("controlnetModel", "offload_mode", "offloadMode"),
    ("prompt", "prompt", "prompt"),
    ("prompt", "negative_prompt", "negativePrompt"),
    ("controlnet", "model_type", "pipelineClass"),
    ("controlnet", "controlnet_variant", "ordinary"),
    ("controlnet", "width", "width"),
    ("controlnet", "height", "height"),
    ("controlnet", "controlnet_conditioning_scale", "conditioningScale"),
    ("denoise", "width", "width"),
    ("denoise", "height", "height"),
    ("denoise", "seed", "seed"),
    ("denoise", "num_inference_steps", "steps"),
    ("denoise", "guidance_scale", "guidanceScale"),
    ("denoise", "strength", "strength"),
)
_MODULAR_SDXL_CONTROL_INPAINT_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -1040, -80),
    ("prompt", "modules.ModularDiffusers.EncodePrompt", -680, -240),
    ("loadImage", "modules.Image.Load", -1040, 200),
    ("loadMask", "modules.Image.Load", -1040, 440),
    ("imageEncode", "modules.ModularDiffusers.ImageEncode", -680, 320),
    ("loadControlImage", "modules.Image.Load", -1040, 680),
    ("controlnetModel", "modules.ModularDiffusers.AutoModelLoader", -680, 760),
    ("controlnet", "modules.ModularDiffusers.Controlnet", -320, 560),
    ("denoise", "modules.ModularDiffusers.Denoise", 80, -80),
    ("decode", "modules.ModularDiffusers.DecodeLatents", 440, -80),
    ("preview", "modules.Image.Preview", 800, -80),
)
_MODULAR_SDXL_CONTROL_INPAINT_GRAPH_EDGES = (
    ("controlnetModel", "model", "models", "controlnet"),
    ("models", "text_encoders", "prompt", "text_encoders"),
    ("models", "unet_out", "denoise", "unet"),
    ("models", "scheduler", "denoise", "scheduler"),
    ("models", "vae_out", "imageEncode", "vae"),
    ("models", "vae_out", "denoise", "vae"),
    ("models", "vae_out", "decode", "vae"),
    ("loadImage", "image", "imageEncode", "image"),
    ("loadMask", "image", "imageEncode", "mask_image"),
    ("loadControlImage", "image", "controlnet", "control_image"),
    ("controlnetModel", "model", "controlnet", "controlnet"),
    ("prompt", "embeddings", "denoise", "embeddings"),
    ("imageEncode", "image_latents", "denoise", "image_latents"),
    ("imageEncode", "mask", "denoise", "mask"),
    ("imageEncode", "masked_image_latents", "denoise", "masked_image_latents"),
    ("imageEncode", "route_state_out", "denoise", "route_state_in"),
    ("controlnet", "controlnet_bundle", "denoise", "controlnet_bundle"),
    ("denoise", "latents", "decode", "latents"),
    ("denoise", "route_state_out", "decode", "route_state_in"),
    ("decode", "images", "preview", "image"),
)
_MODULAR_SDXL_CONTROL_INPAINT_GRAPH_BINDINGS = (
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
    ("loadMask", "file", "maskImage"),
    ("loadMask", "alpha_channel", "removeAlpha"),
    ("imageEncode", "seed", "seed"),
    ("loadControlImage", "file", "controlImage"),
    ("loadControlImage", "alpha_channel", "alphaMode"),
    ("controlnetModel", "model_type", "kind"),
    ("controlnetModel", "model_id", "repo"),
    ("controlnetModel", "dtype", "dtype"),
    ("controlnetModel", "subfolder", "empty"),
    ("controlnetModel", "variant", "fp16"),
    ("controlnetModel", "trust_remote_code", "false"),
    ("controlnetModel", "revision", "revision"),
    ("controlnetModel", "device", "device"),
    ("controlnetModel", "auto_offload", "autoOffload"),
    ("controlnetModel", "offload_mode", "offloadMode"),
    ("prompt", "prompt", "prompt"),
    ("prompt", "negative_prompt", "negativePrompt"),
    ("controlnet", "model_type", "pipelineClass"),
    ("controlnet", "controlnet_variant", "ordinary"),
    ("controlnet", "width", "width"),
    ("controlnet", "height", "height"),
    ("controlnet", "controlnet_conditioning_scale", "conditioningScale"),
    ("denoise", "width", "width"),
    ("denoise", "height", "height"),
    ("denoise", "seed", "seed"),
    ("denoise", "num_inference_steps", "steps"),
    ("denoise", "guidance_scale", "guidanceScale"),
    ("denoise", "strength", "strength"),
)


def _modular_sdxl_conditioned_graph(*, route: str, control: str | None, ip_adapter: bool):
    """Compose reviewed SDXL graph fragments without hiding upstream blocks."""

    if route not in {"text2image", "image2image", "inpainting"}:
        raise ValueError(f"Unknown SDXL route {route!r}.")
    if control not in {None, "ordinary", "union"}:
        raise ValueError(f"Unknown SDXL ControlNet route {control!r}.")
    if control:
        selected = {
            "text2image": (
                _MODULAR_SDXL_CONTROL_GRAPH_ROLES,
                _MODULAR_SDXL_CONTROL_GRAPH_EDGES,
                _MODULAR_SDXL_CONTROL_GRAPH_BINDINGS,
            ),
            "image2image": (
                _MODULAR_SDXL_CONTROL_EDIT_GRAPH_ROLES,
                _MODULAR_SDXL_CONTROL_EDIT_GRAPH_EDGES,
                _MODULAR_SDXL_CONTROL_EDIT_GRAPH_BINDINGS,
            ),
            "inpainting": (
                _MODULAR_SDXL_CONTROL_INPAINT_GRAPH_ROLES,
                _MODULAR_SDXL_CONTROL_INPAINT_GRAPH_EDGES,
                _MODULAR_SDXL_CONTROL_INPAINT_GRAPH_BINDINGS,
            ),
        }[route]
    else:
        selected = {
            "text2image": (
                _MODULAR_TEXT_TO_IMAGE_GRAPH_ROLES,
                _MODULAR_TEXT_TO_IMAGE_GRAPH_EDGES,
                _MODULAR_TEXT_TO_IMAGE_GRAPH_BINDINGS,
            ),
            "image2image": (
                _MODULAR_SDXL_EDIT_GRAPH_ROLES,
                _MODULAR_SDXL_EDIT_GRAPH_EDGES,
                _MODULAR_SDXL_EDIT_GRAPH_BINDINGS,
            ),
            "inpainting": (
                _MODULAR_SDXL_INPAINT_GRAPH_ROLES,
                _MODULAR_SDXL_INPAINT_GRAPH_EDGES,
                _MODULAR_SDXL_INPAINT_GRAPH_BINDINGS,
            ),
        }[route]
    roles, edges, bindings = map(tuple, selected)
    if control:
        source_map = {
            "kind": "controlnetKind",
            "repo": "controlnetRepo",
            "revision": "controlnetRevision",
            "fp16": "controlnetWeightVariant",
            "ordinary": "controlnetRouteVariant",
        }
        bindings = tuple((role, field, source_map.get(source, source)) for role, field, source in bindings)
        bindings += (("controlnetModel", "component_class", "controlnetLoadClass"),)
        if control == "union":
            bindings += (("controlnet", "control_mode", "controlMode"),)
    if ip_adapter:
        roles += (
            ("loadIPAdapterImage", "modules.Image.Load", -1040, 920),
            ("guider", "modules.ModularDiffusers.Guider", -320, 920),
            ("ipAdapter", "modules.ModularDiffusers.IPAdapter", 80, 720),
        )
        edges += (
            ("loadIPAdapterImage", "image", "ipAdapter", "ip_adapter_image"),
            ("models", "unet_out", "ipAdapter", "unet"),
            ("guider", "guider_out", "ipAdapter", "guider"),
            ("guider", "guider_out", "denoise", "guider"),
            ("ipAdapter", "ip_adapter", "denoise", "ip_adapter"),
        )
        bindings += (
            ("loadIPAdapterImage", "file", "ipAdapterImage"),
            ("loadIPAdapterImage", "alpha_channel", "alphaMode"),
            ("guider", "guider", "classifierFreeGuidance"),
            ("guider", "guidance_scale", "guidanceScale"),
            ("ipAdapter", "adapter_model", "ipAdapterRepo"),
            ("ipAdapter", "adapter_revision", "ipAdapterRevision"),
            ("ipAdapter", "adapter_weight_name", "ipAdapterWeightName"),
            ("ipAdapter", "adapter_scale", "ipAdapterScale"),
        )
    return roles, edges, bindings


_MODULAR_SDXL_REMAINING_GRAPHS = {
    (route, control, ip_adapter): _modular_sdxl_conditioned_graph(
        route=route,
        control=control,
        ip_adapter=ip_adapter,
    )
    for route, control, ip_adapter in (
        ("text2image", "union", False),
        ("image2image", "union", False),
        ("inpainting", "union", False),
        ("text2image", None, True),
        ("image2image", None, True),
        ("inpainting", None, True),
        ("text2image", "ordinary", True),
        ("image2image", "ordinary", True),
        ("inpainting", "ordinary", True),
        ("text2image", "union", True),
        ("image2image", "union", True),
        ("inpainting", "union", True),
    )
}
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
    ("models", "revision", "defaultRevision"),
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
_MODULAR_LAYERED_GRAPH_EDGES = tuple(edge for edge in _MODULAR_EDIT_GRAPH_EDGES if edge[1] != "route_state_out")
_MODULAR_LAYERED_GRAPH_BINDINGS = (
    ("models", "model_type", "pipelineClass"),
    ("models", "repo_id", "artifact"),
    ("models", "revision", "defaultRevision"),
    ("models", "dtype", "dtype"),
    ("models", "device", "device"),
    ("models", "auto_offload", "autoOffload"),
    ("models", "offload_mode", "offloadMode"),
    ("models", "trust_remote_code", "false"),
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "addAlpha"),
    ("prompt", "prompt", "prompt"),
    ("prompt", "negative_prompt", "negativePrompt"),
    # Upstream Qwen Image Layered consumes one shared source-resolution
    # argument in both its text/image prompt preprocessing and VAE image
    # encoder blocks. Keep both ordinary nodes synchronized from one public
    # BlockDefinitionV2 value.
    ("prompt", "resolution", "resolution"),
    ("prompt", "max_sequence_length", "maxSequenceLength"),
    ("imageEncode", "resolution", "resolution"),
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
    ("controlnetModel", "model", "models", "controlnet"),
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
    ("models", "revision", "defaultRevision"),
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
    ("prompt", "max_sequence_length", "maxSequenceLength"),
    ("controlnet", "model_type", "pipelineClass"),
    ("controlnet", "width", "width"),
    ("controlnet", "height", "height"),
    ("controlnet", "seed", "seed"),
    ("controlnet", "controlnet_conditioning_scale", "conditioningScale"),
    ("controlnet", "control_guidance_start", "controlGuidanceStart"),
    ("controlnet", "control_guidance_end", "controlGuidanceEnd"),
    ("denoise", "width", "width"),
    ("denoise", "height", "height"),
    ("denoise", "seed", "seed"),
    ("denoise", "num_inference_steps", "steps"),
    ("denoise", "guidance_scale", "guidanceScale"),
    ("denoise", "strength", "strength"),
)
_MODULAR_QWEN_IMAGE_TO_IMAGE_GRAPH_EDGES = tuple(
    edge for edge in _MODULAR_SDXL_EDIT_GRAPH_EDGES if edge != ("models", "vae_out", "denoise", "vae")
)
_MODULAR_QWEN_TEXT_TO_IMAGE_GRAPH_EDGES = tuple(
    edge
    for edge in _MODULAR_TEXT_TO_IMAGE_GRAPH_EDGES
    if edge != ("models", "vae_out", "denoise", "vae")
)
_MODULAR_QWEN_TEXT_TO_IMAGE_GRAPH_BINDINGS = _MODULAR_TEXT_TO_IMAGE_GRAPH_BINDINGS + (
    ("models", "reviewed_variant", "modelVariant"),
    ("prompt", "max_sequence_length", "maxSequenceLength"),
)
_MODULAR_QWEN_IMAGE_TO_IMAGE_GRAPH_BINDINGS = _MODULAR_SDXL_EDIT_GRAPH_BINDINGS + (
    ("prompt", "max_sequence_length", "maxSequenceLength"),
    ("imageEncode", "height", "height"),
    ("imageEncode", "width", "width"),
)
_MODULAR_QWEN_INPAINT_GRAPH_EDGES = tuple(
    edge
    for edge in _MODULAR_SDXL_INPAINT_GRAPH_EDGES
    if edge
    not in {
        ("models", "vae_out", "denoise", "vae"),
        ("imageEncode", "mask", "denoise", "mask"),
        ("imageEncode", "masked_image_latents", "denoise", "masked_image_latents"),
    }
)
_MODULAR_QWEN_EDIT_INPAINT_GRAPH_EDGES = _MODULAR_QWEN_INPAINT_GRAPH_EDGES + (
    ("loadImage", "image", "prompt", "image"),
)
_MODULAR_QWEN_INPAINT_GRAPH_BINDINGS = _MODULAR_SDXL_INPAINT_GRAPH_BINDINGS + (
    ("prompt", "max_sequence_length", "maxSequenceLength"),
    ("imageEncode", "height", "height"),
    ("imageEncode", "width", "width"),
    ("imageEncode", "padding_mask_crop", "paddingMaskCrop"),
)
_MODULAR_QWEN_EDIT_INPAINT_GRAPH_BINDINGS = tuple(
    binding
    for binding in _MODULAR_QWEN_INPAINT_GRAPH_BINDINGS
    if binding
    not in {
        ("imageEncode", "height", "height"),
        ("imageEncode", "width", "width"),
        ("denoise", "height", "height"),
        ("denoise", "width", "width"),
        ("prompt", "max_sequence_length", "maxSequenceLength"),
    }
)
_MODULAR_QWEN_CONTROL_EDIT_GRAPH_EDGES = (
    ("controlnetModel", "model", "models", "controlnet"),
    ("models", "text_encoders", "prompt", "text_encoders"),
    ("models", "unet_out", "denoise", "unet"),
    ("models", "scheduler", "denoise", "scheduler"),
    ("models", "vae_out", "imageEncode", "vae"),
    ("models", "vae_out", "controlnet", "vae"),
    ("models", "vae_out", "decode", "vae"),
    ("loadImage", "image", "imageEncode", "image"),
    ("loadControlImage", "image", "controlnet", "control_image"),
    ("controlnetModel", "model", "controlnet", "controlnet"),
    ("prompt", "embeddings", "denoise", "embeddings"),
    ("imageEncode", "image_latents", "denoise", "image_latents"),
    ("imageEncode", "route_state_out", "controlnet", "route_state_in"),
    ("controlnet", "controlnet_bundle", "denoise", "controlnet_bundle"),
    ("controlnet", "route_state_out", "denoise", "route_state_in"),
    ("denoise", "latents", "decode", "latents"),
    ("denoise", "route_state_out", "decode", "route_state_in"),
    ("decode", "images", "preview", "image"),
)
_MODULAR_QWEN_CONTROL_EDIT_GRAPH_BINDINGS = tuple(
    ("loadControlImage" if role == "loadImage" else role, field, source)
    for role, field, source in _MODULAR_CONTROL_GRAPH_BINDINGS
) + (
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "alphaMode"),
    ("imageEncode", "height", "height"),
    ("imageEncode", "width", "width"),
    ("imageEncode", "seed", "seed"),
)
_MODULAR_QWEN_CONTROL_INPAINT_GRAPH_EDGES = _MODULAR_QWEN_CONTROL_EDIT_GRAPH_EDGES + (
    ("loadMask", "image", "imageEncode", "mask_image"),
)
_MODULAR_QWEN_CONTROL_INPAINT_GRAPH_BINDINGS = _MODULAR_QWEN_CONTROL_EDIT_GRAPH_BINDINGS + (
    ("loadMask", "file", "maskImage"),
    ("loadMask", "alpha_channel", "removeAlpha"),
    ("imageEncode", "padding_mask_crop", "paddingMaskCrop"),
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
_PAG_CONDITIONED_CONTROL_GRAPH_BINDINGS = _CONDITIONED_CONTROL_GRAPH_BINDINGS + (
    ("diffusersImageControl", "pag_scale", "pagScale"),
    ("diffusersImageControl", "pag_adaptive_scale", "pagAdaptiveScale"),
)
_QWEN_DIRECT_CONTROL_GRAPH_BINDINGS = tuple(
    item for item in _CONTROL_GRAPH_BINDINGS if item[:2] != ("diffusersImageControl", "strength")
) + (
    ("diffusersImagePipeline", "revision", "defaultRevision"),
    ("diffusersImagePipeline", "conditioning_kind", "kind"),
    ("diffusersImagePipeline", "conditioning_model_id", "repo"),
    ("diffusersImagePipeline", "conditioning_revision", "revision"),
    ("diffusersImageControl", "conditioning_scale", "conditioningScale"),
    ("diffusersImageControl", "control_guidance_start", "controlGuidanceStart"),
    ("diffusersImageControl", "control_guidance_end", "controlGuidanceEnd"),
)
_LAYER_DECOMPOSITION_GRAPH_ROLES = (
    ("diffusersQuantization", "modules.DiffusersRuntime.PipelineQuantizationConfigV2", -1280, -80),
    ("diffusersRecipe", "modules.DiffusersRuntime.DiffusersExecutionRecipe", -900, -80),
    ("diffusersImagePipeline", "modules.DiffusersImage.LoadPipeline", -520, -80),
    ("loadImage", "modules.Image.Load", -520, 300),
    ("diffusersImageLayerDecompose", "modules.DiffusersImage.LayerDecompose", -120, -80),
    ("preview", "modules.Image.Preview", 980, -80),
)
_LAYER_DECOMPOSITION_GRAPH_EDGES = (
    ("diffusersQuantization", "quantization_config", "diffusersRecipe", "quantization_config"),
    ("diffusersRecipe", "execution_recipe", "diffusersImagePipeline", "execution_recipe"),
    ("diffusersImagePipeline", "pipeline", "diffusersImageLayerDecompose", "pipeline"),
    ("loadImage", "image", "diffusersImageLayerDecompose", "image"),
    ("diffusersImageLayerDecompose", "images", "preview", "image"),
)
_LAYER_DECOMPOSITION_GRAPH_BINDINGS = _IMAGE_PIPELINE_BINDINGS + (
    ("diffusersImagePipeline", "revision", "defaultRevision"),
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "addAlpha"),
    ("diffusersImageLayerDecompose", "prompt", "prompt"),
    ("diffusersImageLayerDecompose", "negative_prompt", "negativePrompt"),
    ("diffusersImageLayerDecompose", "seed", "seed"),
    ("diffusersImageLayerDecompose", "num_inference_steps", "steps"),
    ("diffusersImageLayerDecompose", "guidance_scale", "guidanceScale"),
    ("diffusersImageLayerDecompose", "max_sequence_length", "maxSequenceLength"),
    ("diffusersImageLayerDecompose", "layers", "layers"),
    ("diffusersImageLayerDecompose", "resolution", "resolution"),
    ("diffusersImageLayerDecompose", "cfg_normalize", "cfgNormalize"),
    ("diffusersImageLayerDecompose", "use_en_prompt", "useEnglishPrompt"),
    ("diffusersImageLayerDecompose", "output_type", "outputType"),
)
_CONTROL_EDIT_GRAPH_ROLES = (
    ("diffusersQuantization", "modules.DiffusersRuntime.PipelineQuantizationConfigV2", -1280, -80),
    ("diffusersRecipe", "modules.DiffusersRuntime.DiffusersExecutionRecipe", -900, -80),
    ("diffusersImagePipeline", "modules.DiffusersImage.LoadPipeline", -520, -80),
    ("loadImage", "modules.Image.Load", -900, 300),
    ("loadControlImage", "modules.Image.Load", -900, 560),
    ("diffusersImageControlEdit", "modules.DiffusersImage.ControlEdit", -120, -80),
    ("preview", "modules.Image.Preview", 980, -80),
)
_CONTROL_EDIT_GRAPH_EDGES = (
    ("diffusersQuantization", "quantization_config", "diffusersRecipe", "quantization_config"),
    ("diffusersRecipe", "execution_recipe", "diffusersImagePipeline", "execution_recipe"),
    ("diffusersImagePipeline", "pipeline", "diffusersImageControlEdit", "pipeline"),
    ("loadImage", "image", "diffusersImageControlEdit", "image"),
    ("loadControlImage", "image", "diffusersImageControlEdit", "control_image"),
    ("diffusersImageControlEdit", "images", "preview", "image"),
)
_CONDITIONED_CONTROL_EDIT_GRAPH_ROLES = (
    ("diffusersQuantization", "modules.DiffusersRuntime.PipelineQuantizationConfigV2", -1280, -80),
    ("diffusersRecipe", "modules.DiffusersRuntime.DiffusersExecutionRecipe", -900, -80),
    ("diffusersImagePipeline", "modules.DiffusersImage.LoadPipeline", -520, -80),
    ("loadImage", "modules.Image.Load", -900, 300),
    ("loadControlImage", "modules.Image.Load", -900, 560),
    ("controlPreprocessor", "modules.ImageFilters.Canny", -520, 560),
    ("diffusersImageControlEdit", "modules.DiffusersImage.ControlEdit", -120, -80),
    ("preview", "modules.Image.Preview", 980, -80),
)
_CONDITIONED_CONTROL_EDIT_GRAPH_EDGES = (
    ("diffusersQuantization", "quantization_config", "diffusersRecipe", "quantization_config"),
    ("diffusersRecipe", "execution_recipe", "diffusersImagePipeline", "execution_recipe"),
    ("diffusersImagePipeline", "pipeline", "diffusersImageControlEdit", "pipeline"),
    ("loadImage", "image", "diffusersImageControlEdit", "image"),
    ("loadControlImage", "image", "controlPreprocessor", "image"),
    ("controlPreprocessor", "output", "diffusersImageControlEdit", "control_image"),
    ("diffusersImageControlEdit", "images", "preview", "image"),
)
_CONTROL_EDIT_GRAPH_BINDINGS = _IMAGE_PIPELINE_BINDINGS + (
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "alphaMode"),
    ("loadControlImage", "file", "controlImage"),
    ("loadControlImage", "alpha_channel", "alphaMode"),
    ("diffusersImageControlEdit", "prompt", "prompt"),
    ("diffusersImageControlEdit", "width", "width"),
    ("diffusersImageControlEdit", "height", "height"),
    ("diffusersImageControlEdit", "seed", "seed"),
    ("diffusersImageControlEdit", "num_inference_steps", "steps"),
    ("diffusersImageControlEdit", "guidance_scale", "guidanceScale"),
    ("diffusersImageControlEdit", "strength", "strength"),
    ("diffusersImageControlEdit", "output_type", "outputType"),
    ("diffusersImageControlEdit", "max_sequence_length", "maxSequenceLength"),
)
_CONDITIONED_CONTROL_EDIT_GRAPH_BINDINGS = _CONTROL_EDIT_GRAPH_BINDINGS + (
    ("diffusersImagePipeline", "revision", "defaultRevision"),
    ("diffusersImagePipeline", "conditioning_kind", "kind"),
    ("diffusersImagePipeline", "conditioning_model_id", "repo"),
    ("diffusersImagePipeline", "conditioning_revision", "revision"),
    ("controlPreprocessor", "low_threshold", "cannyLowThreshold"),
    ("controlPreprocessor", "high_threshold", "cannyHighThreshold"),
    ("controlPreprocessor", "device", "device"),
    ("diffusersImageControlEdit", "negative_prompt", "negativePrompt"),
    ("diffusersImageControlEdit", "conditioning_scale", "conditioningScale"),
)
_PAG_CONDITIONED_CONTROL_EDIT_GRAPH_BINDINGS = _CONDITIONED_CONTROL_EDIT_GRAPH_BINDINGS + (
    ("diffusersImageControlEdit", "pag_scale", "pagScale"),
    ("diffusersImageControlEdit", "pag_adaptive_scale", "pagAdaptiveScale"),
)
_CONTROL_INPAINT_GRAPH_ROLES = (
    ("diffusersQuantization", "modules.DiffusersRuntime.PipelineQuantizationConfigV2", -1280, -80),
    ("diffusersRecipe", "modules.DiffusersRuntime.DiffusersExecutionRecipe", -900, -80),
    ("diffusersImagePipeline", "modules.DiffusersImage.LoadPipeline", -520, -80),
    ("loadImage", "modules.Image.Load", -900, 300),
    ("loadMask", "modules.Image.Load", -900, 560),
    ("loadControlImage", "modules.Image.Load", -900, 820),
    ("diffusersImageControlInpaint", "modules.DiffusersImage.ControlInpaint", -120, -80),
    ("preview", "modules.Image.Preview", 980, -80),
)
_CONTROL_INPAINT_GRAPH_EDGES = (
    ("diffusersQuantization", "quantization_config", "diffusersRecipe", "quantization_config"),
    ("diffusersRecipe", "execution_recipe", "diffusersImagePipeline", "execution_recipe"),
    ("diffusersImagePipeline", "pipeline", "diffusersImageControlInpaint", "pipeline"),
    ("loadImage", "image", "diffusersImageControlInpaint", "image"),
    ("loadMask", "image", "diffusersImageControlInpaint", "mask_image"),
    ("loadControlImage", "image", "diffusersImageControlInpaint", "control_image"),
    ("diffusersImageControlInpaint", "images", "preview", "image"),
)
_CONDITIONED_CONTROL_INPAINT_GRAPH_ROLES = (
    ("diffusersQuantization", "modules.DiffusersRuntime.PipelineQuantizationConfigV2", -1280, -80),
    ("diffusersRecipe", "modules.DiffusersRuntime.DiffusersExecutionRecipe", -900, -80),
    ("diffusersImagePipeline", "modules.DiffusersImage.LoadPipeline", -520, -80),
    ("loadImage", "modules.Image.Load", -900, 300),
    ("loadMask", "modules.Image.Load", -900, 560),
    ("loadControlImage", "modules.Image.Load", -900, 820),
    ("controlPreprocessor", "modules.ImageFilters.Canny", -520, 820),
    ("diffusersImageControlInpaint", "modules.DiffusersImage.ControlInpaint", -120, -80),
    ("preview", "modules.Image.Preview", 980, -80),
)
_CONDITIONED_CONTROL_INPAINT_GRAPH_EDGES = (
    ("diffusersQuantization", "quantization_config", "diffusersRecipe", "quantization_config"),
    ("diffusersRecipe", "execution_recipe", "diffusersImagePipeline", "execution_recipe"),
    ("diffusersImagePipeline", "pipeline", "diffusersImageControlInpaint", "pipeline"),
    ("loadImage", "image", "diffusersImageControlInpaint", "image"),
    ("loadMask", "image", "diffusersImageControlInpaint", "mask_image"),
    ("loadControlImage", "image", "controlPreprocessor", "image"),
    ("controlPreprocessor", "output", "diffusersImageControlInpaint", "control_image"),
    ("diffusersImageControlInpaint", "images", "preview", "image"),
)
_CONTROL_INPAINT_GRAPH_BINDINGS = _IMAGE_PIPELINE_BINDINGS + (
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "alphaMode"),
    ("loadMask", "file", "maskImage"),
    ("loadMask", "alpha_channel", "removeAlpha"),
    ("loadControlImage", "file", "controlImage"),
    ("loadControlImage", "alpha_channel", "alphaMode"),
    ("diffusersImageControlInpaint", "prompt", "prompt"),
    ("diffusersImageControlInpaint", "width", "width"),
    ("diffusersImageControlInpaint", "height", "height"),
    ("diffusersImageControlInpaint", "seed", "seed"),
    ("diffusersImageControlInpaint", "num_inference_steps", "steps"),
    ("diffusersImageControlInpaint", "guidance_scale", "guidanceScale"),
    ("diffusersImageControlInpaint", "strength", "strength"),
    ("diffusersImageControlInpaint", "output_type", "outputType"),
    ("diffusersImageControlInpaint", "max_sequence_length", "maxSequenceLength"),
)
_CONDITIONED_CONTROL_INPAINT_GRAPH_BINDINGS = _CONTROL_INPAINT_GRAPH_BINDINGS + (
    ("diffusersImagePipeline", "revision", "defaultRevision"),
    ("diffusersImagePipeline", "conditioning_kind", "kind"),
    ("diffusersImagePipeline", "conditioning_model_id", "repo"),
    ("diffusersImagePipeline", "conditioning_revision", "revision"),
    ("controlPreprocessor", "low_threshold", "cannyLowThreshold"),
    ("controlPreprocessor", "high_threshold", "cannyHighThreshold"),
    ("controlPreprocessor", "device", "device"),
    ("diffusersImageControlInpaint", "negative_prompt", "negativePrompt"),
    ("diffusersImageControlInpaint", "conditioning_scale", "conditioningScale"),
)
_PAG_CONDITIONED_CONTROL_INPAINT_GRAPH_BINDINGS = _CONDITIONED_CONTROL_INPAINT_GRAPH_BINDINGS + (
    ("diffusersImageControlInpaint", "pag_scale", "pagScale"),
    ("diffusersImageControlInpaint", "pag_adaptive_scale", "pagAdaptiveScale"),
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
_QWEN_DIRECT_EDIT_GRAPH_BINDINGS = tuple(
    item
    for item in _EDIT_GRAPH_BINDINGS
    if item[:2]
    not in {
        ("diffusersImageEdit", "strength"),
        ("diffusersImageEdit", "reference_strength"),
    }
) + (("diffusersImagePipeline", "revision", "defaultRevision"),)
_SDXL_EDIT_GRAPH_BINDINGS = _EDIT_GRAPH_BINDINGS + (("diffusersImagePipeline", "revision", "defaultRevision"),)
_PAG_EDIT_GRAPH_BINDINGS = _SDXL_EDIT_GRAPH_BINDINGS + (
    ("diffusersImageEdit", "pag_scale", "pagScale"),
    ("diffusersImageEdit", "pag_adaptive_scale", "pagAdaptiveScale"),
)
_LCM_EDIT_GRAPH_BINDINGS = tuple(
    item
    for item in _SDXL_EDIT_GRAPH_BINDINGS
    if item[0] != "diffusersImageEdit"
    or item[1]
    in {
        "prompt",
        "seed",
        "num_inference_steps",
        "guidance_scale",
        "strength",
        "output_type",
    }
)
_SD15_PAG_EDIT_GRAPH_BINDINGS = tuple(
    item
    for item in _PAG_EDIT_GRAPH_BINDINGS
    if item[0] != "diffusersImageEdit"
    or item[1]
    in {
        "prompt",
        "negative_prompt",
        "seed",
        "num_inference_steps",
        "guidance_scale",
        "strength",
        "pag_scale",
        "pag_adaptive_scale",
        "output_type",
    }
)
_SDXL_INSTRUCT_EDIT_GRAPH_BINDINGS = tuple(
    item for item in _SDXL_EDIT_GRAPH_BINDINGS if item[:2] != ("diffusersImageEdit", "reference_strength")
) + (("diffusersImageEdit", "image_guidance_scale", "conditioningScale"),)
_OMNIGEN_GRAPH_BINDINGS = tuple(
    item
    for item in _SDXL_GRAPH_BINDINGS
    if item[:2]
    not in {
        ("diffusersImageGenerate", "negative_prompt"),
        ("diffusersImageGenerate", "strength"),
        ("diffusersImageGenerate", "max_sequence_length"),
    }
)
_OMNIGEN_EDIT_GRAPH_BINDINGS = tuple(
    item
    for item in _SDXL_EDIT_GRAPH_BINDINGS
    if item[:2]
    not in {
        ("diffusersImageEdit", "negative_prompt"),
        ("diffusersImageEdit", "strength"),
        ("diffusersImageEdit", "reference_strength"),
        ("diffusersImageEdit", "max_sequence_length"),
    }
) + (("diffusersImageEdit", "image_guidance_scale", "conditioningScale"),)
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
_SDXL_INPAINT_GRAPH_BINDINGS = _INPAINT_GRAPH_BINDINGS + (("diffusersImagePipeline", "revision", "defaultRevision"),)
_PAG_INPAINT_GRAPH_BINDINGS = _SDXL_INPAINT_GRAPH_BINDINGS + (
    ("diffusersImageInpaint", "pag_scale", "pagScale"),
    ("diffusersImageInpaint", "pag_adaptive_scale", "pagAdaptiveScale"),
)
_SD15_PAG_INPAINT_GRAPH_BINDINGS = tuple(
    item
    for item in _PAG_INPAINT_GRAPH_BINDINGS
    if item[0] != "diffusersImageInpaint"
    or item[1]
    in {
        "prompt",
        "negative_prompt",
        "width",
        "height",
        "seed",
        "num_inference_steps",
        "guidance_scale",
        "strength",
        "pag_scale",
        "pag_adaptive_scale",
        "output_type",
    }
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
_OUTPAINT_GRAPH_ROLES = (
    ("diffusersQuantization", "modules.DiffusersRuntime.PipelineQuantizationConfigV2", -1280, -80),
    ("diffusersRecipe", "modules.DiffusersRuntime.DiffusersExecutionRecipe", -900, -80),
    ("diffusersImagePipeline", "modules.DiffusersImage.LoadPipeline", -520, -80),
    ("loadImage", "modules.Image.Load", -520, 300),
    ("outpaintCanvas", "modules.DiffusersImage.OutpaintCanvas", -520, 300),
    ("diffusersImageInpaint", "modules.DiffusersImage.Inpaint", -120, -80),
    ("preview", "modules.Image.Preview", 980, -80),
)
_OUTPAINT_GRAPH_EDGES = (
    ("diffusersQuantization", "quantization_config", "diffusersRecipe", "quantization_config"),
    ("diffusersRecipe", "execution_recipe", "diffusersImagePipeline", "execution_recipe"),
    ("diffusersImagePipeline", "pipeline", "diffusersImageInpaint", "pipeline"),
    ("loadImage", "image", "outpaintCanvas", "image"),
    ("outpaintCanvas", "canvas", "diffusersImageInpaint", "image"),
    ("outpaintCanvas", "mask_image", "diffusersImageInpaint", "mask_image"),
    ("diffusersImageInpaint", "images", "preview", "image"),
)
_OUTPAINT_GRAPH_BINDINGS = tuple(item for item in _INPAINT_GRAPH_BINDINGS if item[0] != "loadMask") + (
    ("outpaintCanvas", "width", "width"),
    ("outpaintCanvas", "height", "height"),
    ("outpaintCanvas", "left", "outpaintLeft"),
    ("outpaintCanvas", "right", "outpaintRight"),
    ("outpaintCanvas", "top", "outpaintTop"),
    ("outpaintCanvas", "bottom", "outpaintBottom"),
    ("outpaintCanvas", "overlap", "outpaintOverlap"),
    ("outpaintCanvas", "feather", "outpaintFeather"),
    ("outpaintCanvas", "fill_color", "outpaintFillColor"),
)


def _direct_inpaint_bindings(*, outpaint: bool, unsupported_params: frozenset[str]) -> tuple:
    bindings = _OUTPAINT_GRAPH_BINDINGS if outpaint else _INPAINT_GRAPH_BINDINGS
    return tuple(
        item for item in bindings if item[0] != "diffusersImageInpaint" or item[1] not in unsupported_params
    ) + (("diffusersImagePipeline", "revision", "defaultRevision"),)


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
_I2V_GRAPH_ROLES = _VIDEO_GRAPH_ROLES + (("loadImage", "modules.Image.Load", -520, 300),)
_I2V_GRAPH_EDGES = _VIDEO_GRAPH_EDGES + (("loadImage", "image", "wanGenerate", "reference_images"),)
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
    (role, param, "outpaintMaskGrow0")
    if role == "alignMaskVideo" and param == "grow_pixels"
    else (role, param, source)
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
_ANIMATEDIFF_CONTROL_GRAPH_ROLES = _VIDEO_GRAPH_ROLES + (
    ("loadControlVideo", "modules.Video.Load", -520, 260),
    (
        "normalizeControlVideo",
        "modules.VideoConditioning.Normalize",
        -160,
        260,
    ),
    (
        "controlPreprocessor",
        "modules.VideoConditioning.EdgePreprocessor",
        220,
        260,
    ),
)
_ANIMATEDIFF_CONTROL_GRAPH_EDGES = _VIDEO_GRAPH_EDGES + (
    ("loadControlVideo", "video", "normalizeControlVideo", "video"),
    (
        "normalizeControlVideo",
        "output",
        "controlPreprocessor",
        "video",
    ),
    ("controlPreprocessor", "output", "wanGenerate", "control_video"),
)
_ANIMATEDIFF_CONTROL_V2V_GRAPH_ROLES = _V2V_GRAPH_ROLES + (
    ("loadControlVideo", "modules.Video.Load", -520, 520),
    (
        "normalizeControlVideo",
        "modules.VideoConditioning.Normalize",
        -160,
        520,
    ),
    (
        "controlPreprocessor",
        "modules.VideoConditioning.EdgePreprocessor",
        220,
        520,
    ),
)
_ANIMATEDIFF_CONTROL_V2V_GRAPH_EDGES = _V2V_GRAPH_EDGES + (
    ("loadControlVideo", "video", "normalizeControlVideo", "video"),
    (
        "normalizeControlVideo",
        "output",
        "controlPreprocessor",
        "video",
    ),
    ("controlPreprocessor", "output", "wanGenerate", "control_video"),
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
_ANIMATEDIFF_DIRECT_GRAPH_BINDINGS = tuple(
    (
        role,
        param,
        "motionAdapterRepo"
        if (role, param) == ("wanPipeline", "motion_adapter_id")
        else "motionAdapterRevision"
        if (role, param) == ("wanPipeline", "motion_adapter_revision")
        else source,
    )
    for role, param, source in _ANIMATEDIFF_GRAPH_BINDINGS
)
_ANIMATEDIFF_PAG_GRAPH_BINDINGS = _ANIMATEDIFF_DIRECT_GRAPH_BINDINGS + (
    ("wanGenerate", "pag_scale", "pagScale"),
    ("wanGenerate", "pag_adaptive_scale", "pagAdaptiveScale"),
)
_ANIMATEDIFF_V2V_GRAPH_BINDINGS = _ANIMATEDIFF_DIRECT_GRAPH_BINDINGS + (
    ("loadVideo", "file", "sourceVideo"),
    ("normalizeVideo", "width", "width"),
    ("normalizeVideo", "height", "height"),
    ("normalizeVideo", "num_frames", "numFrames"),
)
_ANIMATEDIFF_CONTROL_INPUT_GRAPH_BINDINGS = (
    ("loadControlVideo", "file", "controlVideo"),
    ("normalizeControlVideo", "width", "width"),
    ("normalizeControlVideo", "height", "height"),
    ("normalizeControlVideo", "num_frames", "numFrames"),
    (
        "controlPreprocessor",
        "low_threshold",
        "videoCannyLowThreshold100",
    ),
    (
        "controlPreprocessor",
        "high_threshold",
        "videoCannyHighThreshold200",
    ),
)
_ANIMATEDIFF_CONTROL_GRAPH_BINDINGS = _ANIMATEDIFF_DIRECT_GRAPH_BINDINGS + _ANIMATEDIFF_CONTROL_INPUT_GRAPH_BINDINGS
_ANIMATEDIFF_CONTROL_V2V_GRAPH_BINDINGS = _ANIMATEDIFF_V2V_GRAPH_BINDINGS + _ANIMATEDIFF_CONTROL_INPUT_GRAPH_BINDINGS
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
_COGVIDEOX_V2V_GRAPH_BINDINGS = _COGVIDEOX_GRAPH_BINDINGS + (
    ("loadVideo", "file", "sourceVideo"),
    ("normalizeVideo", "width", "width"),
    ("normalizeVideo", "height", "height"),
    ("normalizeVideo", "num_frames", "numFrames"),
)
_SANA_VIDEO_I2V_GRAPH_BINDINGS = tuple(
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
    for role, param, source in _I2V_REVISION_GRAPH_BINDINGS
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
_LTX2_IN_CONTEXT_GRAPH_ROLES = _LTX2_GRAPH_ROLES + (
    ("loadReferenceVideo", "modules.Video.Load", -720, 300),
    ("normalizeReferenceVideo", "modules.VideoConditioning.Normalize", -360, 300),
    ("referencePreprocessor", "modules.VideoConditioning.EdgePreprocessor", 0, 300),
)
_LTX2_IN_CONTEXT_GRAPH_EDGES = _LTX2_GRAPH_EDGES + (
    ("loadReferenceVideo", "video", "normalizeReferenceVideo", "video"),
    ("normalizeReferenceVideo", "output", "referencePreprocessor", "video"),
    ("referencePreprocessor", "output", "wanGenerate", "reference_video"),
)
_LTX2_IN_CONTEXT_GRAPH_BINDINGS = tuple(
    (role, param, "requiredNumFrames") if role == "wanGenerate" and param == "num_frames" else (role, param, source)
    for role, param, source in _LTX2_GRAPH_BINDINGS
) + (
    ("wanPipeline", "ic_lora_id", "repo"),
    ("wanPipeline", "ic_lora_revision", "revision"),
    ("wanPipeline", "ic_lora_weight_name", "adapterWeightName"),
    ("loadReferenceVideo", "file", "referenceVideos"),
    ("normalizeReferenceVideo", "width", "width"),
    ("normalizeReferenceVideo", "height", "height"),
    ("normalizeReferenceVideo", "num_frames", "requiredNumFrames"),
    ("wanGenerate", "reference_strength", "strength"),
    ("wanGenerate", "conditioning_attention_strength", "conditioningScale"),
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
    ("prompt", "max_sequence_length", "maxSequenceLength"),
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
_WAN_IMAGE_TO_VIDEO_GRAPH_ROLES = tuple(
    role for role in _WAN_FLF_GRAPH_ROLES if role[0] != "loadLastImage"
)
_WAN_IMAGE_TO_VIDEO_GRAPH_EDGES = tuple(
    edge
    for edge in _WAN_FLF_GRAPH_EDGES
    if edge[0] != "loadLastImage" and edge[2] != "loadLastImage"
)
_WAN_IMAGE_TO_VIDEO_GRAPH_BINDINGS = tuple(
    binding for binding in _WAN_FLF_GRAPH_BINDINGS if binding[0] != "loadLastImage"
)
_WAN_TEXT_TO_VIDEO_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -900, -80),
    ("prompt", "modules.ModularDiffusers.EncodePrompt", -520, -240),
    ("denoise", "modules.ModularDiffusers.Denoise", -100, -80),
    ("decode", "modules.ModularDiffusers.DecodeLatents", 320, -80),
    ("videoExport", "modules.Video.Export", 720, -80),
)
_WAN_TEXT_TO_VIDEO_GRAPH_EDGES = (
    ("models", "text_encoders", "prompt", "text_encoders"),
    ("models", "unet_out", "denoise", "unet"),
    ("models", "scheduler", "denoise", "scheduler"),
    ("models", "vae_out", "decode", "vae"),
    ("prompt", "embeddings", "denoise", "embeddings"),
    ("denoise", "latents", "decode", "latents"),
    ("decode", "videos", "videoExport", "video"),
)
_WAN_TEXT_TO_VIDEO_GRAPH_BINDINGS = tuple(
    binding
    for binding in _WAN_FLF_GRAPH_BINDINGS
    if binding[0] in {"models", "prompt", "denoise", "decode", "videoExport"}
)
_MODULAR_WHOLE_AUDIO_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -900, -80),
    ("prompt", "modules.ModularDiffusers.WorkflowSemanticGeneration", -500, -80),
    ("denoise", "modules.ModularDiffusers.WorkflowDenoise", -100, -80),
    ("decode", "modules.ModularDiffusers.WorkflowDecodeAudio", 300, -80),
    ("audioExport", "modules.Audio.Export", 700, -80),
)
_MODULAR_WHOLE_AUDIO_GRAPH_EDGES = (
    ("models", "pipeline_components", "prompt", "pipeline_components"),
    ("models", "pipeline_components", "denoise", "pipeline_components"),
    ("models", "pipeline_components", "decode", "pipeline_components"),
    ("prompt", "state_out", "denoise", "state_in"),
    ("denoise", "state_out", "decode", "state_in"),
    ("decode", "audio", "audioExport", "audio"),
)
_MODULAR_WHOLE_AUDIO_GRAPH_BINDINGS = (
    ("models", "model_type", "pipelineClass"),
    ("models", "repo_id", "artifact"),
    ("models", "revision", "defaultRevision"),
    ("models", "dtype", "dtype"),
    ("models", "device", "device"),
    ("models", "auto_offload", "autoOffload"),
    ("models", "offload_mode", "offloadMode"),
    ("models", "trust_remote_code", "false"),
    ("prompt", "pipeline_class", "pipelineClass"),
    ("prompt", "workflow_id", "defaultWorkflow"),
    ("prompt", "block_path", "semanticGeneratorBlock"),
    ("prompt", "prompt", "prompt"),
    ("prompt", "lyrics", "lyrics"),
    ("prompt", "audio_duration", "audioDuration"),
    ("prompt", "seed", "seed"),
    ("denoise", "pipeline_class", "pipelineClass"),
    ("denoise", "workflow_id", "defaultWorkflow"),
    ("denoise", "block_path", "workflowDenoiseBlock"),
    ("denoise", "num_inference_steps", "steps"),
    ("decode", "pipeline_class", "pipelineClass"),
    ("decode", "workflow_id", "defaultWorkflow"),
    ("decode", "block_path", "workflowDecodeBlock"),
    ("audioExport", "sample_rate", "sampleRate44100"),
)
_MODULAR_WHOLE_IMAGE_TEXT_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -900, -80),
    ("prompt", "modules.ModularDiffusers.WorkflowTextEncode", -500, -80),
    ("denoise", "modules.ModularDiffusers.WorkflowImageDenoise", -100, -80),
    ("decode", "modules.ModularDiffusers.WorkflowDecodeImage", 300, -80),
    ("preview", "modules.Image.Preview", 700, -80),
)
_MODULAR_WHOLE_IMAGE_TEXT_GRAPH_EDGES = (
    ("models", "pipeline_components", "prompt", "pipeline_components"),
    ("models", "pipeline_components", "denoise", "pipeline_components"),
    ("models", "pipeline_components", "decode", "pipeline_components"),
    ("prompt", "state_out", "denoise", "state_in"),
    ("denoise", "state_out", "decode", "state_in"),
    ("decode", "images", "preview", "image"),
)
_MODULAR_WHOLE_IMAGE_EDIT_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -1060, -80),
    ("loadImage", "modules.Image.Load", -720, 260),
    ("prompt", "modules.ModularDiffusers.WorkflowTextEncode", -660, -160),
    ("imageEncode", "modules.ModularDiffusers.WorkflowImageEncode", -240, 260),
    ("denoise", "modules.ModularDiffusers.WorkflowImageDenoise", 180, -80),
    ("decode", "modules.ModularDiffusers.WorkflowDecodeImage", 580, -80),
    ("preview", "modules.Image.Preview", 980, -80),
)
_MODULAR_WHOLE_IMAGE_EDIT_GRAPH_EDGES = (
    ("models", "pipeline_components", "prompt", "pipeline_components"),
    ("models", "pipeline_components", "imageEncode", "pipeline_components"),
    ("models", "pipeline_components", "denoise", "pipeline_components"),
    ("models", "pipeline_components", "decode", "pipeline_components"),
    ("loadImage", "image", "imageEncode", "image"),
    ("prompt", "state_out", "imageEncode", "state_in"),
    ("imageEncode", "state_out", "denoise", "state_in"),
    ("denoise", "state_out", "decode", "state_in"),
    ("decode", "images", "preview", "image"),
)
_MODULAR_WHOLE_IMAGE_COMMON_GRAPH_BINDINGS = (
    ("models", "model_type", "pipelineClass"),
    ("models", "repo_id", "artifact"),
    ("models", "revision", "defaultRevision"),
    ("models", "dtype", "dtype"),
    ("models", "device", "device"),
    ("models", "auto_offload", "autoOffload"),
    ("models", "offload_mode", "offloadMode"),
    ("models", "trust_remote_code", "false"),
    ("prompt", "pipeline_class", "pipelineClass"),
    ("prompt", "workflow_id", "workflowId"),
    ("prompt", "block_path", "workflowTextEncoderBlock"),
    ("prompt", "prompt", "prompt"),
    ("prompt", "negative_prompt", "negativePrompt"),
    ("prompt", "max_sequence_length", "maxSequenceLength"),
    ("denoise", "pipeline_class", "pipelineClass"),
    ("denoise", "workflow_id", "workflowId"),
    ("denoise", "block_path", "workflowDenoiseBlock"),
    ("denoise", "width", "width"),
    ("denoise", "height", "height"),
    ("denoise", "seed", "seed"),
    ("denoise", "num_inference_steps", "steps"),
    ("denoise", "guidance_scale", "guidanceScale"),
    ("denoise", "strength", "strength"),
    ("decode", "pipeline_class", "pipelineClass"),
    ("decode", "workflow_id", "workflowId"),
    ("decode", "block_path", "workflowDecodeBlock"),
)
_MODULAR_WHOLE_IMAGE_TEXT_GRAPH_BINDINGS = _MODULAR_WHOLE_IMAGE_COMMON_GRAPH_BINDINGS
_MODULAR_WHOLE_IMAGE_EDIT_GRAPH_BINDINGS = _MODULAR_WHOLE_IMAGE_COMMON_GRAPH_BINDINGS + (
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "alphaMode"),
    ("imageEncode", "pipeline_class", "pipelineClass"),
    ("imageEncode", "workflow_id", "workflowId"),
    ("imageEncode", "block_path", "workflowImageEncoderBlock"),
    ("imageEncode", "width", "width"),
    ("imageEncode", "height", "height"),
    ("imageEncode", "seed", "seed"),
)
_MODULAR_WHOLE_VIDEO_TEXT_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -900, -80),
    ("prompt", "modules.ModularDiffusers.WorkflowTextEncode", -520, -80),
    ("denoise", "modules.ModularDiffusers.WorkflowVideoDenoise", -120, -80),
    ("decode", "modules.ModularDiffusers.WorkflowDecodeVideo", 280, -80),
    ("videoExport", "modules.Video.Export", 680, -80),
)
_MODULAR_WHOLE_VIDEO_TEXT_GRAPH_EDGES = (
    ("models", "pipeline_components", "prompt", "pipeline_components"),
    ("models", "pipeline_components", "denoise", "pipeline_components"),
    ("models", "pipeline_components", "decode", "pipeline_components"),
    ("prompt", "state_out", "denoise", "state_in"),
    ("denoise", "state_out", "decode", "state_in"),
    ("decode", "video", "videoExport", "video"),
)
_MODULAR_WHOLE_VIDEO_IMAGE_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -1060, -80),
    ("loadImage", "modules.Image.Load", -760, 280),
    ("prompt", "modules.ModularDiffusers.WorkflowTextEncode", -680, -160),
    ("imageEncode", "modules.ModularDiffusers.WorkflowVideoImageEncode", -280, 260),
    ("denoise", "modules.ModularDiffusers.WorkflowVideoDenoise", 140, -80),
    ("decode", "modules.ModularDiffusers.WorkflowDecodeVideo", 540, -80),
    ("videoExport", "modules.Video.Export", 940, -80),
)
_MODULAR_WHOLE_VIDEO_IMAGE_GRAPH_EDGES = (
    ("models", "pipeline_components", "prompt", "pipeline_components"),
    ("models", "pipeline_components", "imageEncode", "pipeline_components"),
    ("models", "pipeline_components", "denoise", "pipeline_components"),
    ("models", "pipeline_components", "decode", "pipeline_components"),
    ("loadImage", "image", "imageEncode", "image"),
    ("prompt", "state_out", "imageEncode", "state_in"),
    ("imageEncode", "state_out", "denoise", "state_in"),
    ("denoise", "state_out", "decode", "state_in"),
    ("decode", "video", "videoExport", "video"),
)
_MODULAR_WHOLE_VIDEO_EDIT_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -1060, -80),
    ("loadVideo", "modules.Video.Load", -760, 280),
    ("prompt", "modules.ModularDiffusers.WorkflowTextEncode", -680, -160),
    ("videoEncode", "modules.ModularDiffusers.WorkflowVideoEncode", -280, 260),
    ("denoise", "modules.ModularDiffusers.WorkflowVideoDenoise", 140, -80),
    ("decode", "modules.ModularDiffusers.WorkflowDecodeVideo", 540, -80),
    ("videoExport", "modules.Video.Export", 940, -80),
)
_MODULAR_WHOLE_VIDEO_EDIT_GRAPH_EDGES = (
    ("models", "pipeline_components", "prompt", "pipeline_components"),
    ("models", "pipeline_components", "videoEncode", "pipeline_components"),
    ("models", "pipeline_components", "denoise", "pipeline_components"),
    ("models", "pipeline_components", "decode", "pipeline_components"),
    ("loadVideo", "video", "videoEncode", "video"),
    ("prompt", "state_out", "videoEncode", "state_in"),
    ("videoEncode", "state_out", "denoise", "state_in"),
    ("denoise", "state_out", "decode", "state_in"),
    ("decode", "video", "videoExport", "video"),
)
_MODULAR_WHOLE_VIDEO_COMMON_GRAPH_BINDINGS = (
    ("models", "model_type", "pipelineClass"),
    ("models", "repo_id", "artifact"),
    ("models", "revision", "defaultRevision"),
    ("models", "dtype", "dtype"),
    ("models", "device", "device"),
    ("models", "auto_offload", "autoOffload"),
    ("models", "offload_mode", "offloadMode"),
    ("models", "trust_remote_code", "false"),
    ("prompt", "pipeline_class", "pipelineClass"),
    ("prompt", "workflow_id", "workflowId"),
    ("prompt", "block_path", "workflowTextEncoderBlock"),
    ("prompt", "prompt", "prompt"),
    ("prompt", "negative_prompt", "negativePrompt"),
    ("prompt", "max_sequence_length", "maxSequenceLength"),
    ("denoise", "pipeline_class", "pipelineClass"),
    ("denoise", "workflow_id", "workflowId"),
    ("denoise", "block_path", "workflowDenoiseBlock"),
    ("denoise", "width", "width"),
    ("denoise", "height", "height"),
    ("denoise", "num_frames", "numFrames"),
    ("denoise", "num_inference_steps", "steps"),
    ("denoise", "pyramid_stage_1_steps", "steps"),
    ("denoise", "pyramid_stage_2_steps", "steps"),
    ("denoise", "pyramid_stage_3_steps", "steps"),
    ("denoise", "guidance_scale", "guidanceScale"),
    ("denoise", "seed", "seed"),
    ("decode", "pipeline_class", "pipelineClass"),
    ("decode", "workflow_id", "workflowId"),
    ("decode", "block_path", "workflowDecodeBlock"),
    ("videoExport", "fps", "fps"),
)
_MODULAR_WHOLE_VIDEO_TEXT_GRAPH_BINDINGS = _MODULAR_WHOLE_VIDEO_COMMON_GRAPH_BINDINGS
_MODULAR_WHOLE_VIDEO_IMAGE_GRAPH_BINDINGS = _MODULAR_WHOLE_VIDEO_COMMON_GRAPH_BINDINGS + (
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "alphaMode"),
    ("imageEncode", "pipeline_class", "pipelineClass"),
    ("imageEncode", "workflow_id", "workflowId"),
    ("imageEncode", "block_path", "workflowImageEncoderBlock"),
    ("imageEncode", "width", "width"),
    ("imageEncode", "height", "height"),
    ("imageEncode", "seed", "seed"),
)
_MODULAR_WHOLE_VIDEO_EDIT_GRAPH_BINDINGS = _MODULAR_WHOLE_VIDEO_COMMON_GRAPH_BINDINGS + (
    ("loadVideo", "file", "sourceVideo"),
    ("videoEncode", "pipeline_class", "pipelineClass"),
    ("videoEncode", "workflow_id", "workflowId"),
    ("videoEncode", "block_path", "workflowImageEncoderBlock"),
    ("videoEncode", "width", "width"),
    ("videoEncode", "height", "height"),
    ("videoEncode", "seed", "seed"),
)
_HUNYUAN_VIDEO_15_T2V_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -900, -80),
    ("prompt", "modules.ModularDiffusers.WorkflowHunyuanVideo15TextEncode", -500, -80),
    ("denoise", "modules.ModularDiffusers.WorkflowHunyuanVideo15Denoise", -100, -80),
    ("decode", "modules.ModularDiffusers.WorkflowHunyuanVideo15Decode", 300, -80),
    ("videoExport", "modules.Video.Export", 700, -80),
)
_HUNYUAN_VIDEO_15_T2V_GRAPH_EDGES = (
    ("models", "pipeline_components", "prompt", "pipeline_components"),
    ("models", "pipeline_components", "denoise", "pipeline_components"),
    ("models", "pipeline_components", "decode", "pipeline_components"),
    ("prompt", "state_out", "denoise", "state_in"),
    ("denoise", "state_out", "decode", "state_in"),
    ("decode", "video", "videoExport", "video"),
)
_HUNYUAN_VIDEO_15_COMMON_GRAPH_BINDINGS = (
    ("models", "model_type", "pipelineClass"),
    ("models", "repo_id", "artifact"),
    ("models", "revision", "defaultRevision"),
    ("models", "dtype", "dtype"),
    ("models", "device", "device"),
    ("models", "auto_offload", "autoOffload"),
    ("models", "offload_mode", "offloadMode"),
    ("models", "trust_remote_code", "false"),
    ("prompt", "pipeline_class", "pipelineClass"),
    ("prompt", "workflow_id", "workflowId"),
    ("prompt", "block_path", "workflowTextEncoderBlock"),
    ("prompt", "prompt", "prompt"),
    ("prompt", "negative_prompt", "negativePrompt"),
    ("prompt", "num_videos_per_prompt", "oneVideo"),
    ("denoise", "pipeline_class", "pipelineClass"),
    ("denoise", "workflow_id", "workflowId"),
    ("denoise", "block_path", "workflowDenoiseBlock"),
    ("denoise", "num_frames", "numFrames"),
    ("denoise", "num_inference_steps", "steps"),
    ("denoise", "guidance_scale", "guidanceScale"),
    ("denoise", "seed", "seed"),
    ("decode", "pipeline_class", "pipelineClass"),
    ("decode", "workflow_id", "workflowId"),
    ("decode", "block_path", "workflowDecodeBlock"),
    ("videoExport", "fps", "fps"),
)
_HUNYUAN_VIDEO_15_T2V_GRAPH_BINDINGS = _HUNYUAN_VIDEO_15_COMMON_GRAPH_BINDINGS + (
    ("denoise", "width", "width"),
    ("denoise", "height", "height"),
)
_HUNYUAN_VIDEO_15_I2V_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -1120, -80),
    ("loadImage", "modules.Image.Load", -820, 300),
    ("prompt", "modules.ModularDiffusers.WorkflowHunyuanVideo15TextEncode", -720, -160),
    ("imageEncode", "modules.ModularDiffusers.WorkflowHunyuanVideo15VaeEncode", -320, 220),
    ("imageEmbeddings", "modules.ModularDiffusers.WorkflowHunyuanVideo15ImageEncode", 80, 220),
    ("denoise", "modules.ModularDiffusers.WorkflowHunyuanVideo15Denoise", 480, -80),
    ("decode", "modules.ModularDiffusers.WorkflowHunyuanVideo15Decode", 880, -80),
    ("videoExport", "modules.Video.Export", 1280, -80),
)
_HUNYUAN_VIDEO_15_I2V_GRAPH_EDGES = (
    ("models", "pipeline_components", "prompt", "pipeline_components"),
    ("models", "pipeline_components", "imageEncode", "pipeline_components"),
    ("models", "pipeline_components", "imageEmbeddings", "pipeline_components"),
    ("models", "pipeline_components", "denoise", "pipeline_components"),
    ("models", "pipeline_components", "decode", "pipeline_components"),
    ("loadImage", "image", "imageEncode", "image"),
    ("prompt", "state_out", "imageEncode", "state_in"),
    ("imageEncode", "state_out", "imageEmbeddings", "state_in"),
    ("imageEmbeddings", "state_out", "denoise", "state_in"),
    ("denoise", "state_out", "decode", "state_in"),
    ("decode", "video", "videoExport", "video"),
)
_HUNYUAN_VIDEO_15_I2V_GRAPH_BINDINGS = _HUNYUAN_VIDEO_15_COMMON_GRAPH_BINDINGS + (
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "alphaMode"),
    ("imageEncode", "pipeline_class", "pipelineClass"),
    ("imageEncode", "workflow_id", "workflowId"),
    ("imageEncode", "block_path", "workflowVaeEncoderBlock"),
    ("imageEncode", "width", "optionalWidth"),
    ("imageEncode", "height", "optionalHeight"),
    ("imageEmbeddings", "pipeline_class", "pipelineClass"),
    ("imageEmbeddings", "workflow_id", "workflowId"),
    ("imageEmbeddings", "block_path", "workflowImageEmbeddingsBlock"),
)
_MINIMAX_H3_T2VA_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -760, -80),
    ("prompt", "modules.ModularDiffusers.WorkflowMiniMaxH3TextEncode", -400, -80),
    ("denoise", "modules.ModularDiffusers.WorkflowMiniMaxH3Denoise", -40, -80),
    ("decode", "modules.ModularDiffusers.WorkflowMiniMaxH3Decode", 320, -80),
    ("videoExport", "modules.Video.ExportWithAudio", 680, -80),
)
_MINIMAX_H3_T2VA_GRAPH_EDGES = (
    ("models", "pipeline_components", "prompt", "pipeline_components"),
    ("models", "pipeline_components", "denoise", "pipeline_components"),
    ("models", "pipeline_components", "decode", "pipeline_components"),
    ("prompt", "state_out", "denoise", "state_in"),
    ("denoise", "state_out", "decode", "state_in"),
    ("decode", "video", "videoExport", "video"),
    ("decode", "audio", "videoExport", "audio"),
)
_MINIMAX_H3_CONDITIONED_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -1080, -80),
    ("beforeEncode", "modules.ModularDiffusers.WorkflowMiniMaxH3BeforeEncode", -720, 200),
    ("prompt", "modules.ModularDiffusers.WorkflowMiniMaxH3TextEncode", -360, -80),
    ("imageEncode", "modules.ModularDiffusers.WorkflowMiniMaxH3VaeEncode", 0, 200),
    ("denoise", "modules.ModularDiffusers.WorkflowMiniMaxH3Denoise", 360, -80),
    ("decode", "modules.ModularDiffusers.WorkflowMiniMaxH3Decode", 720, -80),
    ("videoExport", "modules.Video.ExportWithAudio", 1080, -80),
)
_MINIMAX_H3_CONDITIONED_GRAPH_EDGES = (
    ("models", "pipeline_components", "beforeEncode", "pipeline_components"),
    ("models", "pipeline_components", "prompt", "pipeline_components"),
    ("models", "pipeline_components", "imageEncode", "pipeline_components"),
    ("models", "pipeline_components", "denoise", "pipeline_components"),
    ("models", "pipeline_components", "decode", "pipeline_components"),
    ("beforeEncode", "state_out", "prompt", "state_in"),
    ("prompt", "state_out", "imageEncode", "state_in"),
    ("imageEncode", "state_out", "denoise", "state_in"),
    ("denoise", "state_out", "decode", "state_in"),
    ("decode", "video", "videoExport", "video"),
    ("decode", "audio", "videoExport", "audio"),
)
_MINIMAX_H3_COMMON_GRAPH_BINDINGS = (
    ("models", "model_type", "pipelineClass"),
    ("models", "repo_id", "artifact"),
    ("models", "revision", "defaultRevision"),
    ("models", "workflow_id", "workflowId"),
    ("models", "dtype", "dtype"),
    ("models", "device", "device"),
    ("models", "auto_offload", "autoOffload"),
    ("models", "offload_mode", "offloadMode"),
    ("models", "trust_remote_code", "false"),
    ("prompt", "pipeline_class", "pipelineClass"),
    ("prompt", "workflow_id", "workflowId"),
    ("prompt", "block_path", "workflowTextEncoderBlock"),
    ("prompt", "prompt", "prompt"),
    ("denoise", "pipeline_class", "pipelineClass"),
    ("denoise", "workflow_id", "workflowId"),
    ("denoise", "block_path", "workflowDenoiseBlock"),
    ("denoise", "num_inference_steps", "steps"),
    ("denoise", "seed", "seed"),
    ("decode", "pipeline_class", "pipelineClass"),
    ("decode", "workflow_id", "workflowId"),
    ("decode", "block_path", "workflowDecodeBlock"),
    ("videoExport", "fps", "fps"),
)
_MINIMAX_H3_T2VA_GRAPH_BINDINGS = _MINIMAX_H3_COMMON_GRAPH_BINDINGS + (
    ("denoise", "width", "width"),
    ("denoise", "height", "height"),
    ("denoise", "num_frames", "numFrames"),
)
_MINIMAX_H3_CONDITIONED_COMMON_GRAPH_BINDINGS = _MINIMAX_H3_COMMON_GRAPH_BINDINGS + (
    ("beforeEncode", "pipeline_class", "pipelineClass"),
    ("beforeEncode", "workflow_id", "workflowId"),
    ("beforeEncode", "block_path", "workflowBeforeEncodeBlock"),
    ("beforeEncode", "width", "width"),
    ("beforeEncode", "height", "height"),
    ("imageEncode", "pipeline_class", "pipelineClass"),
    ("imageEncode", "workflow_id", "workflowId"),
    ("imageEncode", "block_path", "workflowVaeEncoderBlock"),
)
_MINIMAX_H3_FL2VA_GRAPH_BINDINGS = _MINIMAX_H3_CONDITIONED_COMMON_GRAPH_BINDINGS + (
    ("beforeEncode", "num_frames", "numFrames"),
    ("beforeEncode", "image", "referenceImages"),
    ("beforeEncode", "last_image", "lastImage"),
)
_MINIMAX_H3_REF2VA_GRAPH_BINDINGS = _MINIMAX_H3_CONDITIONED_COMMON_GRAPH_BINDINGS + (
    ("beforeEncode", "num_frames", "requiredNumFrames"),
    ("beforeEncode", "references", "references"),
)
_COSMOS3_OMNI_TEXT_TO_IMAGE_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -980, -80),
    ("prompt", "modules.ModularDiffusers.WorkflowCosmos3OmniTextEncode", -580, -80),
    ("denoise", "modules.ModularDiffusers.WorkflowCosmos3OmniDenoise", -160, -80),
    ("decode", "modules.ModularDiffusers.WorkflowCosmos3OmniDecode", 260, -80),
    ("afterDecode", "modules.ModularDiffusers.WorkflowCosmos3OmniAfterDecode", 680, 240),
    ("preview", "modules.Image.Preview", 680, -160),
)
_COSMOS3_OMNI_TEXT_TO_IMAGE_GRAPH_EDGES = (
    ("models", "pipeline_components", "prompt", "pipeline_components"),
    ("models", "pipeline_components", "denoise", "pipeline_components"),
    ("models", "pipeline_components", "decode", "pipeline_components"),
    ("models", "pipeline_components", "afterDecode", "pipeline_components"),
    ("prompt", "state_out", "denoise", "state_in"),
    ("denoise", "state_out", "decode", "state_in"),
    ("decode", "state_out", "afterDecode", "state_in"),
    ("decode", "image", "preview", "image"),
)
_COSMOS3_OMNI_TEXT_TO_VIDEO_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -980, -80),
    ("prompt", "modules.ModularDiffusers.WorkflowCosmos3OmniTextEncode", -580, -80),
    ("denoise", "modules.ModularDiffusers.WorkflowCosmos3OmniDenoise", -160, -80),
    ("decode", "modules.ModularDiffusers.WorkflowCosmos3OmniDecode", 260, -80),
    ("afterDecode", "modules.ModularDiffusers.WorkflowCosmos3OmniAfterDecode", 680, 240),
    ("videoExport", "modules.Video.Export", 680, -160),
)
_COSMOS3_OMNI_TEXT_TO_VIDEO_GRAPH_EDGES = (
    ("models", "pipeline_components", "prompt", "pipeline_components"),
    ("models", "pipeline_components", "denoise", "pipeline_components"),
    ("models", "pipeline_components", "decode", "pipeline_components"),
    ("models", "pipeline_components", "afterDecode", "pipeline_components"),
    ("prompt", "state_out", "denoise", "state_in"),
    ("denoise", "state_out", "decode", "state_in"),
    ("decode", "state_out", "afterDecode", "state_in"),
    ("decode", "videos", "videoExport", "video"),
)
_COSMOS3_OMNI_IMAGE_TO_VIDEO_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -1180, -80),
    ("loadImage", "modules.Image.Load", -980, 300),
    ("prompt", "modules.ModularDiffusers.WorkflowCosmos3OmniTextEncode", -780, -160),
    ("imageEncode", "modules.ModularDiffusers.WorkflowCosmos3OmniVaeEncode", -360, 220),
    ("denoise", "modules.ModularDiffusers.WorkflowCosmos3OmniDenoise", 60, -80),
    ("decode", "modules.ModularDiffusers.WorkflowCosmos3OmniDecode", 480, -80),
    ("afterDecode", "modules.ModularDiffusers.WorkflowCosmos3OmniAfterDecode", 900, 240),
    ("videoExport", "modules.Video.Export", 900, -160),
)
_COSMOS3_OMNI_IMAGE_TO_VIDEO_GRAPH_EDGES = (
    ("models", "pipeline_components", "prompt", "pipeline_components"),
    ("models", "pipeline_components", "imageEncode", "pipeline_components"),
    ("models", "pipeline_components", "denoise", "pipeline_components"),
    ("models", "pipeline_components", "decode", "pipeline_components"),
    ("models", "pipeline_components", "afterDecode", "pipeline_components"),
    ("loadImage", "image", "imageEncode", "image"),
    ("prompt", "state_out", "imageEncode", "state_in"),
    ("imageEncode", "state_out", "denoise", "state_in"),
    ("denoise", "state_out", "decode", "state_in"),
    ("decode", "state_out", "afterDecode", "state_in"),
    ("decode", "videos", "videoExport", "video"),
)
_COSMOS3_OMNI_VIDEO_TO_VIDEO_GRAPH_ROLES = tuple(
    (
        role,
        "modules.Video.Load" if role == "loadImage" else node_key,
        x,
        y,
    )
    for role, node_key, x, y in _COSMOS3_OMNI_IMAGE_TO_VIDEO_GRAPH_ROLES
)
_COSMOS3_OMNI_VIDEO_TO_VIDEO_GRAPH_ROLES = tuple(
    ("loadVideo" if role == "loadImage" else role, node_key, x, y)
    for role, node_key, x, y in _COSMOS3_OMNI_VIDEO_TO_VIDEO_GRAPH_ROLES
)
_COSMOS3_OMNI_VIDEO_TO_VIDEO_GRAPH_EDGES = tuple(
    (
        "loadVideo" if source_role == "loadImage" else source_role,
        "video" if source_role == "loadImage" and source_output == "image" else source_output,
        target_role,
        "video" if target_role == "imageEncode" and target_input == "image" else target_input,
    )
    for source_role, source_output, target_role, target_input in _COSMOS3_OMNI_IMAGE_TO_VIDEO_GRAPH_EDGES
)


def _cosmos3_sound_graph_roles(roles):
    return tuple(
        (
            role,
            "modules.Video.ExportWithAudio" if role == "videoExport" else node_key,
            x,
            y,
        )
        for role, node_key, x, y in roles
    )


def _cosmos3_sound_graph_edges(edges):
    return (*edges, ("decode", "audio", "videoExport", "audio"))


_COSMOS3_OMNI_TEXT_TO_VIDEO_WITH_AUDIO_GRAPH_ROLES = _cosmos3_sound_graph_roles(
    _COSMOS3_OMNI_TEXT_TO_VIDEO_GRAPH_ROLES
)
_COSMOS3_OMNI_TEXT_TO_VIDEO_WITH_AUDIO_GRAPH_EDGES = _cosmos3_sound_graph_edges(
    _COSMOS3_OMNI_TEXT_TO_VIDEO_GRAPH_EDGES
)
_COSMOS3_OMNI_IMAGE_TO_VIDEO_WITH_AUDIO_GRAPH_ROLES = _cosmos3_sound_graph_roles(
    _COSMOS3_OMNI_IMAGE_TO_VIDEO_GRAPH_ROLES
)
_COSMOS3_OMNI_IMAGE_TO_VIDEO_WITH_AUDIO_GRAPH_EDGES = _cosmos3_sound_graph_edges(
    _COSMOS3_OMNI_IMAGE_TO_VIDEO_GRAPH_EDGES
)
_COSMOS3_OMNI_VIDEO_TO_VIDEO_WITH_AUDIO_GRAPH_ROLES = _cosmos3_sound_graph_roles(
    _COSMOS3_OMNI_VIDEO_TO_VIDEO_GRAPH_ROLES
)
_COSMOS3_OMNI_VIDEO_TO_VIDEO_WITH_AUDIO_GRAPH_EDGES = _cosmos3_sound_graph_edges(
    _COSMOS3_OMNI_VIDEO_TO_VIDEO_GRAPH_EDGES
)
_COSMOS3_OMNI_TEXT_COMMON_GRAPH_BINDINGS = (
    ("models", "model_type", "pipelineClass"),
    ("models", "repo_id", "artifact"),
    ("models", "revision", "defaultRevision"),
    ("models", "dtype", "dtype"),
    ("models", "device", "device"),
    ("models", "auto_offload", "autoOffload"),
    ("models", "offload_mode", "offloadMode"),
    ("models", "trust_remote_code", "false"),
    ("prompt", "pipeline_class", "pipelineClass"),
    ("prompt", "workflow_id", "workflowId"),
    ("prompt", "block_path", "workflowTextEncoderBlock"),
    ("prompt", "prompt", "prompt"),
    ("prompt", "negative_prompt", "negativePrompt"),
    ("prompt", "width", "width"),
    ("prompt", "height", "height"),
    ("prompt", "fps", "fps"),
    ("denoise", "pipeline_class", "pipelineClass"),
    ("denoise", "workflow_id", "workflowId"),
    ("denoise", "block_path", "workflowDenoiseBlock"),
    ("denoise", "num_inference_steps", "steps"),
    ("denoise", "guidance_scale", "guidanceScale"),
    ("denoise", "seed", "seed"),
    ("decode", "pipeline_class", "pipelineClass"),
    ("decode", "workflow_id", "workflowId"),
    ("decode", "block_path", "workflowDecodeBlock"),
    ("afterDecode", "pipeline_class", "pipelineClass"),
    ("afterDecode", "workflow_id", "workflowId"),
    ("afterDecode", "block_path", "workflowAfterDecodeBlock"),
)
_COSMOS3_OMNI_TEXT_TO_IMAGE_GRAPH_BINDINGS = _COSMOS3_OMNI_TEXT_COMMON_GRAPH_BINDINGS + (
    ("prompt", "num_frames", "oneFrame"),
)
_COSMOS3_OMNI_TEXT_TO_VIDEO_GRAPH_BINDINGS = _COSMOS3_OMNI_TEXT_COMMON_GRAPH_BINDINGS + (
    ("prompt", "num_frames", "numFrames"),
    ("videoExport", "fps", "fps"),
)
_COSMOS3_OMNI_CONDITIONED_GRAPH_BINDINGS = (
    ("imageEncode", "pipeline_class", "pipelineClass"),
    ("imageEncode", "workflow_id", "workflowId"),
    ("imageEncode", "block_path", "workflowVaeEncoderBlock"),
)
_COSMOS3_OMNI_IMAGE_TO_VIDEO_GRAPH_BINDINGS = (
    *_COSMOS3_OMNI_TEXT_TO_VIDEO_GRAPH_BINDINGS,
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "alphaMode"),
    *_COSMOS3_OMNI_CONDITIONED_GRAPH_BINDINGS,
)
_COSMOS3_OMNI_VIDEO_TO_VIDEO_GRAPH_BINDINGS = (
    *_COSMOS3_OMNI_TEXT_TO_VIDEO_GRAPH_BINDINGS,
    ("loadVideo", "file", "sourceVideo"),
    *_COSMOS3_OMNI_CONDITIONED_GRAPH_BINDINGS,
)
_COSMOS3_DISTILLED_T2I_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -900, -80),
    ("prompt", "modules.ModularDiffusers.WorkflowCosmos3DistilledTextEncode", -500, -80),
    ("denoise", "modules.ModularDiffusers.WorkflowCosmos3DistilledDenoise", -100, -80),
    ("decode", "modules.ModularDiffusers.WorkflowCosmos3DistilledDecode", 300, -80),
    ("preview", "modules.Image.Preview", 700, -80),
)
_COSMOS3_DISTILLED_T2I_GRAPH_EDGES = (
    ("models", "pipeline_components", "prompt", "pipeline_components"),
    ("models", "pipeline_components", "denoise", "pipeline_components"),
    ("models", "pipeline_components", "decode", "pipeline_components"),
    ("prompt", "state_out", "denoise", "state_in"),
    ("denoise", "state_out", "decode", "state_in"),
    ("decode", "image", "preview", "image"),
)
_COSMOS3_DISTILLED_I2V_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -1100, -80),
    ("loadImage", "modules.Image.Load", -900, 280),
    ("prompt", "modules.ModularDiffusers.WorkflowCosmos3DistilledTextEncode", -700, -160),
    ("imageEncode", "modules.ModularDiffusers.WorkflowCosmos3DistilledVaeEncode", -300, 200),
    ("denoise", "modules.ModularDiffusers.WorkflowCosmos3DistilledDenoise", 100, -80),
    ("decode", "modules.ModularDiffusers.WorkflowCosmos3DistilledDecode", 500, -80),
    ("videoExport", "modules.Video.Export", 900, -80),
)
_COSMOS3_DISTILLED_I2V_GRAPH_EDGES = (
    ("models", "pipeline_components", "prompt", "pipeline_components"),
    ("models", "pipeline_components", "imageEncode", "pipeline_components"),
    ("models", "pipeline_components", "denoise", "pipeline_components"),
    ("models", "pipeline_components", "decode", "pipeline_components"),
    ("loadImage", "image", "imageEncode", "image"),
    ("prompt", "state_out", "imageEncode", "state_in"),
    ("imageEncode", "state_out", "denoise", "state_in"),
    ("denoise", "state_out", "decode", "state_in"),
    ("decode", "video", "videoExport", "video"),
)
_COSMOS3_DISTILLED_COMMON_GRAPH_BINDINGS = (
    ("models", "model_type", "pipelineClass"),
    ("models", "repo_id", "artifact"),
    ("models", "revision", "defaultRevision"),
    ("models", "dtype", "dtype"),
    ("models", "device", "device"),
    ("models", "auto_offload", "autoOffload"),
    ("models", "offload_mode", "offloadMode"),
    ("models", "trust_remote_code", "false"),
    ("prompt", "pipeline_class", "pipelineClass"),
    ("prompt", "workflow_id", "workflowId"),
    ("prompt", "block_path", "workflowTextEncoderBlock"),
    ("prompt", "prompt", "prompt"),
    ("prompt", "width", "width"),
    ("prompt", "height", "height"),
    ("prompt", "fps", "fps"),
    ("prompt", "use_system_prompt", "true"),
    ("prompt", "add_resolution_template", "true"),
    ("prompt", "add_duration_template", "true"),
    ("denoise", "pipeline_class", "pipelineClass"),
    ("denoise", "workflow_id", "workflowId"),
    ("denoise", "block_path", "workflowDenoiseBlock"),
    ("denoise", "num_inference_steps", "distilledSteps4"),
    ("denoise", "guidance_scale", "distilledGuidance1"),
    ("denoise", "seed", "seed"),
    ("decode", "pipeline_class", "pipelineClass"),
    ("decode", "workflow_id", "workflowId"),
    ("decode", "block_path", "workflowDecodeBlock"),
)
_COSMOS3_DISTILLED_T2I_GRAPH_BINDINGS = _COSMOS3_DISTILLED_COMMON_GRAPH_BINDINGS + (
    ("prompt", "num_frames", "oneFrame"),
)
_COSMOS3_DISTILLED_I2V_GRAPH_BINDINGS = _COSMOS3_DISTILLED_COMMON_GRAPH_BINDINGS + (
    ("prompt", "num_frames", "numFrames"),
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "alphaMode"),
    ("imageEncode", "pipeline_class", "pipelineClass"),
    ("imageEncode", "workflow_id", "workflowId"),
    ("imageEncode", "block_path", "workflowVaeEncoderBlock"),
    ("videoExport", "fps", "fps"),
)
_WAN_ANIMATE_2_GRAPH_ROLES = (
    ("models", "modules.ModularDiffusers.ModelsLoader", -1300, -80),
    ("loadImage", "modules.Image.Load", -1080, 280),
    ("loadPoseVideo", "modules.Video.Load", -1080, 600),
    ("prompt", "modules.ModularDiffusers.WorkflowWanAnimateTextEncode", -920, -120),
    ("imageEmbeddings", "modules.ModularDiffusers.WorkflowWanAnimateImageEncode", -520, -120),
    ("videoEncode", "modules.ModularDiffusers.WorkflowWanAnimateVideoEncode", -120, 180),
    ("imageEncode", "modules.ModularDiffusers.WorkflowWanAnimateVaeEncode", 280, 180),
    ("denoise", "modules.ModularDiffusers.WorkflowWanAnimateDenoise", 680, -80),
    ("decode", "modules.ModularDiffusers.WorkflowWanAnimateDecode", 1080, -80),
    ("videoExport", "modules.Video.Export", 1480, -80),
)
_WAN_ANIMATE_2_GRAPH_EDGES = (
    ("models", "pipeline_components", "prompt", "pipeline_components"),
    ("models", "pipeline_components", "imageEmbeddings", "pipeline_components"),
    ("models", "pipeline_components", "videoEncode", "pipeline_components"),
    ("models", "pipeline_components", "imageEncode", "pipeline_components"),
    ("models", "pipeline_components", "denoise", "pipeline_components"),
    ("models", "pipeline_components", "decode", "pipeline_components"),
    ("loadImage", "image", "imageEmbeddings", "image"),
    ("loadPoseVideo", "video", "videoEncode", "driving_video"),
    ("loadPoseVideo", "fps", "videoEncode", "driving_video_fps"),
    ("prompt", "state_out", "imageEmbeddings", "state_in"),
    ("imageEmbeddings", "state_out", "videoEncode", "state_in"),
    ("videoEncode", "state_out", "imageEncode", "state_in"),
    ("imageEncode", "state_out", "denoise", "state_in"),
    ("denoise", "state_out", "decode", "state_in"),
    ("decode", "video", "videoExport", "video"),
)
_WAN_ANIMATE_2_GRAPH_BINDINGS = (
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
    ("loadPoseVideo", "file", "poseVideo"),
    ("prompt", "pipeline_class", "pipelineClass"),
    ("prompt", "workflow_id", "workflowId"),
    ("prompt", "prompt", "prompt"),
    ("prompt", "negative_prompt", "negativePrompt"),
    ("prompt", "prompt_ref", "promptRef"),
    ("prompt", "max_sequence_length", "maxSequenceLength"),
    ("imageEmbeddings", "pipeline_class", "pipelineClass"),
    ("imageEmbeddings", "workflow_id", "workflowId"),
    ("imageEmbeddings", "width", "width"),
    ("imageEmbeddings", "height", "height"),
    ("videoEncode", "pipeline_class", "pipelineClass"),
    ("videoEncode", "workflow_id", "workflowId"),
    ("videoEncode", "fps", "fps"),
    ("videoEncode", "segment_frame_length", "segmentFrameLength"),
    ("videoEncode", "prev_segment_conditioning_frames", "previousConditioningFrames"),
    ("imageEncode", "pipeline_class", "pipelineClass"),
    ("imageEncode", "workflow_id", "workflowId"),
    ("denoise", "pipeline_class", "pipelineClass"),
    ("denoise", "workflow_id", "workflowId"),
    ("denoise", "num_inference_steps", "steps"),
    ("denoise", "guidance_scale", "guidanceScale"),
    ("denoise", "seed", "seed"),
    ("decode", "pipeline_class", "pipelineClass"),
    ("decode", "workflow_id", "workflowId"),
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
    ("diffusersRecipe", "attention_backend", "nativeMath"),
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
    (
        role,
        param,
        "false"
        if role == "diffusersRecipe" and param in {"vae_slicing", "vae_tiling"}
        else "sampleRate24000"
        if param == "sample_rate"
        else "nativeMath"
        if role == "diffusersRecipe" and param == "attention_backend"
        else source,
    )
    for role, param, source in _STABLE_AUDIO_GRAPH_BINDINGS
    if param != "num_waveforms"
)
_AUDIO_LDM2_GRAPH_BINDINGS = tuple(
    (
        role,
        param,
        "sampleRate16000" if param == "sample_rate" else "numWaveforms3" if param == "num_waveforms" else source,
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
_THREE_D_IMAGE_GRAPH_ROLES = _THREE_D_GRAPH_ROLES + (("loadImage", "modules.Image.Load", -520, 260),)
_THREE_D_IMAGE_GRAPH_EDGES = _THREE_D_GRAPH_EDGES + (
    ("loadImage", "image", "diffusersThreeDGenerate", "reference_images"),
)
_THREE_D_IMAGE_GRAPH_BINDINGS = tuple(
    binding
    for binding in _THREE_D_GRAPH_BINDINGS
    if not (binding[0] == "diffusersThreeDGenerate" and binding[1] == "prompt")
) + (
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "alphaMode"),
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
_BINDING_SOURCES = frozenset({"executionProfileId"}) | frozenset(
    item[2]
    for item in (
        *_GRAPH_BINDINGS,
        *_SDXL_GRAPH_BINDINGS,
        *_PAG_GRAPH_BINDINGS,
        *_PERCEPTION_GRAPH_BINDINGS,
        *_SPEECH_TRANSCRIPTION_GRAPH_BINDINGS,
        *_SPEECH_TRANSLATION_GRAPH_BINDINGS,
        *_CTC_SPEECH_GRAPH_BINDINGS,
        *_TRANSFORMERS_TEXT_GRAPH_BINDINGS,
        *_TRANSFORMERS_IMAGE_TEXT_GRAPH_BINDINGS,
        *_TRANSFORMERS_ANY_TO_ANY_TEXT_GRAPH_BINDINGS,
        *_TRANSFORMERS_ANY_TO_ANY_IMAGE_TEXT_GRAPH_BINDINGS,
        *_TRANSFORMERS_ANY_TO_ANY_IMAGE_GRAPH_BINDINGS,
        *_MODULAR_TEXT_TO_IMAGE_GRAPH_BINDINGS,
        *_MODULAR_QWEN_TEXT_TO_IMAGE_GRAPH_BINDINGS,
        *_MODULAR_SDXL_EDIT_GRAPH_BINDINGS,
        *_MODULAR_SDXL_INPAINT_GRAPH_BINDINGS,
        *_MODULAR_SDXL_CONTROL_GRAPH_BINDINGS,
        *_MODULAR_SDXL_CONTROL_EDIT_GRAPH_BINDINGS,
        *_MODULAR_SDXL_CONTROL_INPAINT_GRAPH_BINDINGS,
        *(
            binding
            for _roles, _edges, conditioned_bindings in _MODULAR_SDXL_REMAINING_GRAPHS.values()
            for binding in conditioned_bindings
        ),
        *_SDXL_EDIT_GRAPH_BINDINGS,
        *_MODULAR_EDIT_GRAPH_BINDINGS,
        *_MODULAR_LAYERED_GRAPH_BINDINGS,
        *_MODULAR_CONTROL_GRAPH_BINDINGS,
        *_MODULAR_QWEN_IMAGE_TO_IMAGE_GRAPH_BINDINGS,
        *_MODULAR_QWEN_INPAINT_GRAPH_BINDINGS,
        *_MODULAR_QWEN_EDIT_INPAINT_GRAPH_BINDINGS,
        *_MODULAR_QWEN_CONTROL_EDIT_GRAPH_BINDINGS,
        *_MODULAR_QWEN_CONTROL_INPAINT_GRAPH_BINDINGS,
        *_CONTROL_GRAPH_BINDINGS,
        *_CONDITIONED_CONTROL_GRAPH_BINDINGS,
        *_QWEN_DIRECT_CONTROL_GRAPH_BINDINGS,
        *_LAYER_DECOMPOSITION_GRAPH_BINDINGS,
        *_EDIT_GRAPH_BINDINGS,
        *_QWEN_DIRECT_EDIT_GRAPH_BINDINGS,
        *_INPAINT_GRAPH_BINDINGS,
        *_QWEN_OUTPAINT_GRAPH_BINDINGS,
        *_OUTPAINT_GRAPH_BINDINGS,
        *_VIDEO_GRAPH_BINDINGS,
        *_WAN_VACE_GRAPH_BINDINGS,
        *_I2V_GRAPH_BINDINGS,
        *_V2V_GRAPH_BINDINGS,
        *_VACE_INPAINT_GRAPH_BINDINGS,
        *_VACE_OUTPAINT_GRAPH_BINDINGS,
        *_VACE_CONTROL_GRAPH_BINDINGS,
        *_ANIMATEDIFF_PAG_GRAPH_BINDINGS,
        *_ANIMATEDIFF_V2V_GRAPH_BINDINGS,
        *_ANIMATEDIFF_CONTROL_GRAPH_BINDINGS,
        *_ANIMATEDIFF_CONTROL_V2V_GRAPH_BINDINGS,
        *_COGVIDEOX_V2V_GRAPH_BINDINGS,
        *_LTX_T2V_GRAPH_BINDINGS,
        *_LTX_I2V_GRAPH_BINDINGS,
        *_LTX_V2V_GRAPH_BINDINGS,
        *_WAN_ANIMATE_GRAPH_BINDINGS,
        *_WAN_REPLACE_GRAPH_BINDINGS,
        *_WAN_ANIMATE_2_GRAPH_BINDINGS,
        *_LTX_LONG_GRAPH_BINDINGS,
        *_LTX2_GRAPH_BINDINGS,
        *_LTX2_I2V_GRAPH_BINDINGS,
        *_LTX2_V2V_GRAPH_BINDINGS,
        *_LTX2_IN_CONTEXT_GRAPH_BINDINGS,
        *_FRAMEPACK_GRAPH_BINDINGS,
        *_WAN_TEXT_TO_VIDEO_GRAPH_BINDINGS,
        *_WAN_IMAGE_TO_VIDEO_GRAPH_BINDINGS,
        *_WAN_FLF_GRAPH_BINDINGS,
        *_MODULAR_WHOLE_AUDIO_GRAPH_BINDINGS,
        *_MODULAR_WHOLE_IMAGE_TEXT_GRAPH_BINDINGS,
        *_MODULAR_WHOLE_IMAGE_EDIT_GRAPH_BINDINGS,
        *_MODULAR_WHOLE_VIDEO_TEXT_GRAPH_BINDINGS,
        *_MODULAR_WHOLE_VIDEO_IMAGE_GRAPH_BINDINGS,
        *_MODULAR_WHOLE_VIDEO_EDIT_GRAPH_BINDINGS,
        *_MINIMAX_H3_T2VA_GRAPH_BINDINGS,
        *_MINIMAX_H3_FL2VA_GRAPH_BINDINGS,
        *_MINIMAX_H3_REF2VA_GRAPH_BINDINGS,
        *_HUNYUAN_VIDEO_15_T2V_GRAPH_BINDINGS,
        *_HUNYUAN_VIDEO_15_I2V_GRAPH_BINDINGS,
        *_COSMOS3_OMNI_TEXT_TO_IMAGE_GRAPH_BINDINGS,
        *_COSMOS3_OMNI_TEXT_TO_VIDEO_GRAPH_BINDINGS,
        *_COSMOS3_OMNI_IMAGE_TO_VIDEO_GRAPH_BINDINGS,
        *_COSMOS3_OMNI_VIDEO_TO_VIDEO_GRAPH_BINDINGS,
        *_COSMOS3_DISTILLED_T2I_GRAPH_BINDINGS,
        *_COSMOS3_DISTILLED_I2V_GRAPH_BINDINGS,
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
) | {"conditionImages", "referenceAudio", "referenceVideos"}

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
    "expert_quantization_modes": ("bnb_4bit",),
}
_MODULAR_QWEN_EDIT_PROFILE = {
    "id": "qwen-edit:modular",
    "model_type": "QwenImageEditModularPipeline",
    "modes": ("edit_image", "modular_inpainting"),
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
}
_MODULAR_QWEN_EDIT_INPAINT_AUTO_REQUIREMENTS = {
    "supportedTasks": ["modular_inpainting"],
    "defaultRepo": "Qwen/Qwen-Image-Edit",
    "qualityDefaults": {
        "width": 768,
        "height": 768,
        "steps": 24,
        "guidanceScale": 4,
        "strength": 0.9,
    },
    "minimum": {
        "accelerator": "cuda",
        "vramBytes": 24 * _GIB,
        "systemRamBytes": 40 * _GIB,
        "diskFreeBytes": 30 * _GIB,
    },
    "recommended": {
        "accelerator": "cuda",
        "vramBytes": 32 * _GIB,
        "systemRamBytes": 48 * _GIB,
        "diskFreeBytes": 40 * _GIB,
    },
    "fullResidency": {
        "accelerator": "cuda",
        "vramBytes": 64 * _GIB,
        "systemRamBytes": 64 * _GIB,
    },
    "supportedOffloadModes": [
        OFFLOAD_MODE_NONE,
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ],
    "requiredPackages": ["diffusers", "transformers", "accelerate", "torch"],
    "guardedReason": (
        "Qwen Image Edit Modular inpainting retains its mask and overlay continuation inside the sealed route state "
        "and is planned independently for each Cluster instance."
    ),
}
_MODULAR_SDXL_PROFILE = {
    "id": "sdxl-base:modular",
    "model_type": "StableDiffusionXLModularPipeline",
    "modes": (
        "text_to_image",
        "edit_image",
        "inpaint",
        "control_image",
        "control_edit_image",
        "control_inpaint",
        *_SDXL_UNION_MODES,
        *_SDXL_IP_ADAPTER_MODES,
        *_SDXL_IP_ADAPTER_CONTROL_MODES,
        *_SDXL_IP_ADAPTER_UNION_MODES,
    ),
    "loader_module": "modules.ModularDiffusers",
    "loader_action": "ModelsLoader",
    "execution_path": "modular-diffusers",
    "pipeline_class": "StableDiffusionXLModularPipeline",
    "default_repo": SDXL_BASE_REPO,
    "fallback_repo": None,
    "quantizable_components": ("unet", "text_encoder", "text_encoder_2"),
    "default_quantized_components": (),
    "supported_offload_modes": (
        OFFLOAD_MODE_NONE,
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ),
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 30,
    "live_proof": False,
    "compatible_repos": (),
}
_MODULAR_FLUX_PROFILE = {
    "id": "flux-dev:modular",
    "model_type": "FluxModularPipeline",
    "modes": ("text_to_image", "image_to_image"),
    "loader_module": "modules.ModularDiffusers",
    "loader_action": "ModelsLoader",
    "execution_path": "modular-diffusers",
    "pipeline_class": "FluxModularPipeline",
    "default_repo": FLUX_DEV_REPO,
    "fallback_repo": None,
    "quantizable_components": ("transformer", "text_encoder_2"),
    "default_quantized_components": ("transformer",),
    "supported_offload_modes": (
        OFFLOAD_MODE_NONE,
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ),
    "retry_offload_modes": (OFFLOAD_MODE_GROUP_DISK,),
    "max_low_memory_side": 768,
    "max_low_memory_steps": 20,
    "live_proof": False,
    "compatible_repos": (FLUX_DEV_FP8_REPO,),
}
_MODULAR_FLUX_KONTEXT_PROFILE = {
    "id": "flux-kontext:modular",
    "model_type": "FluxKontextModularPipeline",
    "modes": ("text_to_image", "edit_image"),
    "loader_module": "modules.ModularDiffusers",
    "loader_action": "ModelsLoader",
    "execution_path": "modular-diffusers",
    "pipeline_class": "FluxKontextModularPipeline",
    "default_repo": FLUX_KONTEXT_REPO,
    "fallback_repo": None,
    "quantizable_components": ("transformer", "text_encoder_2"),
    "default_quantized_components": ("transformer",),
    "supported_offload_modes": (
        OFFLOAD_MODE_NONE,
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ),
    "retry_offload_modes": (OFFLOAD_MODE_GROUP_DISK,),
    "max_low_memory_side": 768,
    "max_low_memory_steps": 20,
    "live_proof": False,
    "compatible_repos": (FLUX_KONTEXT_NVFP4_REPO,),
}
_MODULAR_FLUX2_KLEIN_PROFILE = {
    "id": "flux2-klein:modular",
    "model_type": "Flux2KleinModularPipeline",
    "modes": ("text_to_image", "edit_image"),
    "loader_module": "modules.ModularDiffusers",
    "loader_action": "ModelsLoader",
    "execution_path": "modular-diffusers",
    "pipeline_class": "Flux2KleinModularPipeline",
    "default_repo": FLUX2_KLEIN_REPO,
    "fallback_repo": None,
    "quantizable_components": ("transformer", "text_encoder"),
    "default_quantized_components": ("transformer",),
    "supported_offload_modes": (
        OFFLOAD_MODE_NONE,
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ),
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK),
    "max_low_memory_side": 768,
    "max_low_memory_steps": 4,
    "live_proof": False,
    "compatible_repos": (),
}
_MODULAR_FLUX2_KLEIN_BASE_PROFILE = {
    "id": "flux2-klein-base:modular",
    "model_type": "Flux2KleinBaseModularPipeline",
    "modes": ("text_to_image", "edit_image"),
    "loader_module": "modules.ModularDiffusers",
    "loader_action": "ModelsLoader",
    "execution_path": "modular-diffusers",
    "pipeline_class": "Flux2KleinBaseModularPipeline",
    "default_repo": FLUX2_KLEIN_BASE_REPO,
    "fallback_repo": None,
    "quantizable_components": ("transformer", "text_encoder"),
    "default_quantized_components": ("transformer",),
    "supported_offload_modes": (
        OFFLOAD_MODE_NONE,
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ),
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK),
    "max_low_memory_side": 768,
    "max_low_memory_steps": 24,
    "live_proof": False,
    "compatible_repos": (),
}
_MODULAR_Z_IMAGE_PROFILE = {
    "id": "z-image:modular",
    "model_type": "ZImageModularPipeline",
    "modes": ("modular_text_to_image", "modular_image_to_image"),
    "loader_module": "modules.ModularDiffusers",
    "loader_action": "ModelsLoader",
    "execution_path": "modular-diffusers",
    "pipeline_class": "ZImageModularPipeline",
    "default_repo": Z_IMAGE_REPO,
    "fallback_repo": None,
    "quantizable_components": ("transformer", "text_encoder"),
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
}
_MODULAR_Z_IMAGE_AUTO_REQUIREMENTS = {
    "supportedTasks": ["modular_text_to_image", "modular_image_to_image"],
    "defaultRepo": Z_IMAGE_REPO,
    "qualityDefaults": {
        "width": 1024,
        "height": 1024,
        "steps": 8,
        "guidanceScale": 1,
        "maxSequenceLength": 512,
    },
    "minimum": {
        "accelerator": "gpu_or_cpu",
        "vramBytes": 0,
        "systemRamBytes": 8 * _GIB,
        "diskFreeBytes": 12 * _GIB,
    },
    "recommended": {
        "accelerator": "gpu",
        "vramBytes": 8 * _GIB,
        "systemRamBytes": 16 * _GIB,
        "diskFreeBytes": 20 * _GIB,
    },
    "fullResidency": _HIGH_MEMORY_FULL_RESIDENCY,
    "supportedOffloadModes": [
        OFFLOAD_MODE_NONE,
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ],
    "requiredPackages": ["diffusers", "transformers", "accelerate", "torch"],
    "guardedReason": (
        "Z-Image Modular workflows are planned per Cluster instance and remain unpublished until their "
        "collapsed/expanded execution equivalence is reviewed."
    ),
}
_MODULAR_FLUX_AUTO_REQUIREMENTS = {
    "supportedTasks": ["text_to_image", "image_to_image"],
    "defaultRepo": FLUX_DEV_REPO,
    "preferredLowerMemoryRepo": FLUX_DEV_FP8_REPO,
    "qualityDefaults": {
        "width": 1024,
        "height": 1024,
        "steps": 28,
        "guidanceScale": 3.5,
        "maxSequenceLength": 512,
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
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
        OFFLOAD_MODE_NONE,
    ],
    "requiredPackages": ["diffusers", "transformers", "accelerate", "torch"],
    "guardedReason": (
        "FLUX Modular Diffusers uses one independently planned Cluster instance; changing its workflow or "
        "artifact invalidates that instance's component and resource receipts without affecting other Clusters."
    ),
}
_MODULAR_FLUX_KONTEXT_AUTO_REQUIREMENTS = {
    "supportedTasks": ["text_to_image", "edit_image"],
    "defaultRepo": FLUX_KONTEXT_REPO,
    "qualityDefaults": {
        "width": 1024,
        "height": 1024,
        "steps": 28,
        "guidanceScale": 2.5,
        "maxSequenceLength": 512,
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
    "supportedOffloadModes": [
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
        OFFLOAD_MODE_NONE,
    ],
    "requiredPackages": ["diffusers", "transformers", "accelerate", "torch"],
    "guardedReason": (
        "FLUX Kontext Modular workflows preserve the exact upstream optional image-latent dispatch inside one "
        "independently planned Cluster instance."
    ),
}
_MODULAR_FLUX2_KLEIN_AUTO_REQUIREMENTS = {
    "supportedTasks": ["text_to_image", "edit_image"],
    "defaultRepo": FLUX2_KLEIN_REPO,
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
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ],
    "requiredPackages": ["diffusers", "transformers", "accelerate", "torch"],
    "guardedReason": (
        "FLUX.2 Klein Modular workflows retain the pinned distilled constructor configuration and are planned "
        "per Cluster instance."
    ),
}
_MODULAR_FLUX2_KLEIN_BASE_AUTO_REQUIREMENTS = {
    "supportedTasks": ["text_to_image", "edit_image"],
    "defaultRepo": FLUX2_KLEIN_BASE_REPO,
    "qualityDefaults": {
        "width": 1024,
        "height": 1024,
        "steps": 50,
        "guidanceScale": 4,
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
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ],
    "requiredPackages": ["diffusers", "transformers", "accelerate", "torch"],
    "guardedReason": (
        "FLUX.2 Klein Base keeps its non-distilled 50-step workflow and classifier-free guider configuration "
        "inside one independently planned Cluster instance."
    ),
}
_MODULAR_FLUX_CAPABILITY = {
    "modelType": "FluxModularPipeline",
    "label": "FLUX.1 dev (Modular Diffusers)",
    "displayName": "FLUX.1-dev",
    "family": "FLUX Image",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": FLUX_DEV_REPO,
    "downloadFiles": FLUX_DEV_DIFFUSERS_FILES,
    "artifactLabel": "Pinned native Diffusers repository",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 28,
    "recommendedGuidance": 3.5,
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
        "default": OFFLOAD_MODE_GROUP_DISK,
        "lowVram": OFFLOAD_MODE_GROUP_DISK,
        "emergency": OFFLOAD_MODE_GROUP_DISK,
        "modes": list(_MODULAR_FLUX_PROFILE["supported_offload_modes"]),
    },
    "lowVram": {
        "dtype": "bfloat16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_GROUP_DISK,
        "steps": 20,
        "width": 768,
        "height": 768,
    },
    "modes": list(_MODULAR_FLUX_PROFILE["modes"]),
    "modeRequirements": {"image_to_image": {"requiredImages": ["referenceImages"]}},
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(FLUX_DEV_REPO, model_type="FluxModularPipeline")],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "liveProof": False,
    "notes": [
        "The two Cluster workflows share one loader profile and common visual roles while retaining their exact upstream state edges.",
        "Public execution and Auto selection remain disabled pending collapsed/expanded visible-frontend qualification and manual review.",
    ],
}
_MODULAR_FLUX_KONTEXT_CAPABILITY = {
    "modelType": "FluxKontextModularPipeline",
    "label": "FLUX.1 Kontext (Modular Diffusers)",
    "displayName": "FLUX.1-Kontext-dev",
    "family": "FLUX Image",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": FLUX_KONTEXT_REPO,
    "downloadFiles": FLUX_KONTEXT_DIFFUSERS_FILES,
    "artifactLabel": "Pinned native Diffusers repository",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 28,
    "recommendedGuidance": 2.5,
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
        "default": OFFLOAD_MODE_GROUP_DISK,
        "lowVram": OFFLOAD_MODE_GROUP_DISK,
        "emergency": OFFLOAD_MODE_GROUP_DISK,
        "modes": list(_MODULAR_FLUX_KONTEXT_PROFILE["supported_offload_modes"]),
    },
    "lowVram": {
        "dtype": "bfloat16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_GROUP_DISK,
        "steps": 20,
        "width": 768,
        "height": 768,
    },
    "modes": list(_MODULAR_FLUX_KONTEXT_PROFILE["modes"]),
    "modeRequirements": {"edit_image": {"requiredImages": ["referenceImages"]}},
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(FLUX_KONTEXT_REPO, model_type="FluxKontextModularPipeline")
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "liveProof": False,
    "notes": [
        "The text and image-conditioned Cluster workflows use the exact upstream auto-block dispatch.",
        "Public execution and Auto selection remain disabled pending visible-frontend qualification and review.",
    ],
}
_MODULAR_FLUX2_KLEIN_CAPABILITY = {
    "modelType": "Flux2KleinModularPipeline",
    "label": "FLUX.2 Klein 4B (Modular Diffusers)",
    "displayName": "FLUX.2-klein-4B",
    "family": "FLUX Image",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": FLUX2_KLEIN_REPO,
    "downloadFiles": FLUX2_KLEIN_DIFFUSERS_FILES,
    "artifactLabel": "Pinned native Diffusers repository",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 4,
    "recommendedGuidance": 1,
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
        "lowVram": OFFLOAD_MODE_MODEL_CPU,
        "emergency": OFFLOAD_MODE_GROUP_DISK,
        "modes": list(_MODULAR_FLUX2_KLEIN_PROFILE["supported_offload_modes"]),
    },
    "lowVram": {
        "dtype": "bfloat16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_MODEL_CPU,
        "steps": 4,
        "width": 768,
        "height": 768,
    },
    "modes": list(_MODULAR_FLUX2_KLEIN_PROFILE["modes"]),
    "modeRequirements": {"edit_image": {"requiredImages": ["referenceImages"]}},
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(FLUX2_KLEIN_REPO, model_type="Flux2KleinModularPipeline")
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "liveProof": False,
    "notes": [
        "The Cluster loader seals the upstream is_distilled=true constructor configuration.",
        "Public execution and Auto selection remain disabled pending visible-frontend qualification and review.",
    ],
}
_MODULAR_FLUX2_KLEIN_BASE_CAPABILITY = {
    "modelType": "Flux2KleinBaseModularPipeline",
    "label": "FLUX.2 Klein Base 4B (Modular Diffusers)",
    "displayName": "FLUX.2-klein-base-4B",
    "family": "FLUX Image",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": FLUX2_KLEIN_BASE_REPO,
    "downloadFiles": FLUX2_KLEIN_BASE_DIFFUSERS_FILES,
    "artifactLabel": "Pinned native Diffusers repository",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 50,
    "recommendedGuidance": 4,
    "guidanceLabel": "Classifier-free guidance",
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
        "lowVram": OFFLOAD_MODE_MODEL_CPU,
        "emergency": OFFLOAD_MODE_GROUP_DISK,
        "modes": list(_MODULAR_FLUX2_KLEIN_BASE_PROFILE["supported_offload_modes"]),
    },
    "lowVram": {
        "dtype": "bfloat16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_MODEL_CPU,
        "steps": 24,
        "width": 768,
        "height": 768,
    },
    "modes": list(_MODULAR_FLUX2_KLEIN_BASE_PROFILE["modes"]),
    "modeRequirements": {"edit_image": {"requiredImages": ["referenceImages"]}},
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(
            FLUX2_KLEIN_BASE_REPO,
            model_type="Flux2KleinBaseModularPipeline",
        )
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "liveProof": False,
    "notes": [
        "This is the official non-distilled Base workflow; it is not aliased to the 4-step distilled Klein model.",
        "Public execution and Auto selection remain disabled pending visible-frontend qualification and review.",
    ],
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
    "modes": (
        "modular_text_to_image",
        "control_image",
        "image_to_image",
        "inpainting",
        "control_edit_image",
        "control_inpaint",
    ),
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
    "compatible_repos": ("Qwen/Qwen-Image",),
}
_MODULAR_QWEN_IMAGE_AUTO_REQUIREMENTS = {
    "supportedTasks": [
        "modular_text_to_image",
        "image_to_image",
        "inpainting",
        "control_edit_image",
        "control_inpaint",
    ],
    "defaultRepo": "Qwen/Qwen-Image-2512",
    "qualityDefaults": {
        "width": 768,
        "height": 768,
        "steps": 28,
        "guidanceScale": 4,
        "maxSequenceLength": 512,
    },
    "minimum": {
        "accelerator": "cuda",
        "vramBytes": 16 * _GIB,
        "systemRamBytes": 32 * _GIB,
        "diskFreeBytes": 20 * _GIB,
    },
    "recommended": {
        "accelerator": "cuda",
        "vramBytes": 24 * _GIB,
        "systemRamBytes": 48 * _GIB,
        "diskFreeBytes": 35 * _GIB,
    },
    "fullResidency": {
        "accelerator": "cuda",
        "vramBytes": 80 * _GIB,
        "systemRamBytes": 64 * _GIB,
    },
    "onLoadQuantization": {
        "accelerator": "cuda",
        "vramBytes": 16 * _GIB,
        "systemRamBytes": 32 * _GIB,
        "diskFreeBytes": 20 * _GIB,
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
    "guardedReason": (
        "Qwen Image Modular workflows use one isolated Cluster instance; ControlNet dependencies are joined only "
        "for the exact selected control workflow."
    ),
}
_QWEN_DIRECT_CONTROL_PROFILE = {
    "id": "qwen-image-controlnet:direct",
    "model_type": "QwenImageControlNetPipeline",
    "modes": ("control_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "QwenImageControlNetPipeline",
    "default_repo": QWEN_IMAGE_2512_REPO,
    "fallback_repo": None,
    "quantizable_components": ("transformer", "text_encoder"),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_SEQUENTIAL_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ),
    "max_low_memory_side": 768,
    "max_low_memory_steps": 28,
    "live_proof": False,
    "compatible_repos": (),
}
_QWEN_DIRECT_LAYERED_PROFILE = {
    "id": "qwen-image-layered:direct",
    "model_type": "QwenImageLayeredPipeline",
    "modes": ("layer_decomposition",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "QwenImageLayeredPipeline",
    "default_repo": "Qwen/Qwen-Image-Layered",
    "fallback_repo": None,
    "quantizable_components": ("transformer", "text_encoder"),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_SEQUENTIAL_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 30,
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


def _flux_redux_profile() -> dict[str, Any]:
    profile = _profile(
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
    )
    profile["modes"] = ("edit_image", "multi_image_reference_edit")
    return profile


_ANIMA_MODULAR_PROFILE = {
    "id": "anima:official-modular-workflow",
    "model_type": "AnimaModularPipeline",
    "modes": ("text_to_image", "image_to_image"),
    "loader_module": "modules.ModularDiffusers",
    "loader_action": "ModelsLoader",
    "execution_path": "modular-diffusers",
    "pipeline_class": "AnimaModularPipeline",
    "default_repo": ANIMA_REPO,
    "fallback_repo": None,
    "quantizable_components": ("transformer", "text_encoder"),
    "default_quantized_components": (),
    "supported_offload_modes": (
        OFFLOAD_MODE_NONE,
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ),
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
    "max_low_memory_side": 768,
    "max_low_memory_steps": 30,
    "live_proof": False,
    "compatible_repos": (),
}
_ANIMA_MODULAR_CAPABILITY = {
    "modelType": "AnimaModularPipeline",
    "label": "Anima (Modular Diffusers)",
    "displayName": "Anima Base v1.0",
    "family": "Anima",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": ANIMA_REPO,
    "downloadFiles": ANIMA_DIFFUSERS_FILES,
    "artifactLabel": "Exact reviewed Modular Diffusers component snapshot",
    "license": "CircleStone Labs Non-Commercial License v1.0",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 40,
    "recommendedGuidance": 4.0,
    "guidanceLabel": "Classifier-free guidance",
    "supportsImageInput": True,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": True,
    "supportsAudioInput": False,
    "outputKind": "image",
    "offloadSupport": {
        "default": OFFLOAD_MODE_MODEL_CPU,
        "lowVram": OFFLOAD_MODE_GROUP_CPU,
        "emergency": OFFLOAD_MODE_GROUP_DISK,
        "modes": [
            OFFLOAD_MODE_NONE,
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        ],
    },
    "lowVram": {
        "dtype": "bfloat16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_GROUP_CPU,
        "steps": 30,
        "width": 768,
        "height": 768,
    },
    "modes": ["text_to_image", "image_to_image"],
    "modeRequirements": {
        "image_to_image": {
            "requiredImages": ["referenceImages"],
            "note": "Requires exactly one source image.",
        }
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(ANIMA_REPO, model_type="AnimaModularPipeline")],
    "autoEligible": False,
    "templateEligible": False,
    "galleryEligible": False,
    "liveProof": False,
    "notes": [
        "Runs the official Anima text_encoder, optional vae_encoder, denoise, and decode Modular blocks.",
        "The CircleStone Labs Non-Commercial License requires revision-bound acknowledgement before install and Run.",
        "Auto and Gallery remain disabled until real execution and manual output review are complete.",
    ],
}


def _hunyuan_video_15_profile(mode, repository, steps):
    slug = "t2v" if mode == "text_to_video" else "i2v-step-distilled"
    return {
        "id": f"hunyuan-video15-{slug}:official-modular-workflow",
        "model_type": "HunyuanVideo15ModularPipeline",
        "modes": (mode,),
        "loader_module": "modules.ModularDiffusers",
        "loader_action": "ModelsLoader",
        "execution_path": "modular-diffusers",
        "pipeline_class": "HunyuanVideo15ModularPipeline",
        "default_repo": repository,
        "fallback_repo": None,
        "quantizable_components": (),
        "default_quantized_components": (),
        # Publisher guidance documents model CPU offload. Other offload and
        # full-residency modes remain closed until measured qualification.
        "supported_offload_modes": (OFFLOAD_MODE_MODEL_CPU,),
        "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU,),
        "max_low_memory_side": None,
        "max_low_memory_steps": steps,
        "live_proof": False,
        "compatible_repos": (),
    }


_HUNYUAN_VIDEO_15_CAPABILITY = {
    "modelType": "HunyuanVideo15ModularPipeline",
    "label": "HunyuanVideo 1.5 (Modular Diffusers)",
    "displayName": "HunyuanVideo 1.5 480p",
    "family": "HunyuanVideo 1.5",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": HUNYUAN_VIDEO_15_T2V_REPO,
    "downloadFiles": HUNYUAN_VIDEO_15_T2V_DIFFUSERS_FILES,
    "artifactSelections": [
        {
            "modes": ["text_to_video"],
            "repo": HUNYUAN_VIDEO_15_T2V_REPO,
            "revision": require_catalog_revision(
                HUNYUAN_VIDEO_15_T2V_REPO,
                model_type="HunyuanVideo15ModularPipeline",
            ),
            "downloadFiles": HUNYUAN_VIDEO_15_T2V_DIFFUSERS_FILES,
            "label": "Exact reviewed 480p text-to-video snapshot",
        },
        {
            "modes": ["image_to_video"],
            "repo": HUNYUAN_VIDEO_15_I2V_REPO,
            "revision": require_catalog_revision(
                HUNYUAN_VIDEO_15_I2V_REPO,
                model_type="HunyuanVideo15ModularPipeline",
            ),
            "downloadFiles": HUNYUAN_VIDEO_15_I2V_DIFFUSERS_FILES,
            "label": "Exact reviewed 480p step-distilled image-to-video snapshot",
        },
    ],
    "artifactLabel": "Exact reviewed immutable Modular Diffusers snapshot",
    "license": "Tencent Hunyuan Community License Agreement",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 848, "height": 480, "aspectRatio": "custom"},
    "recommendedSteps": 50,
    "recommendedGuidance": 6.0,
    "guidanceLabel": "Official HunyuanVideo 1.5 guidance",
    "supportsImageInput": True,
    "supportsVideoInput": False,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": False,
    "supportsAudioInput": False,
    "outputKind": "video",
    "recommendedFrames": 121,
    "recommendedFps": 24,
    "offloadSupport": {
        "default": OFFLOAD_MODE_MODEL_CPU,
        "lowVram": OFFLOAD_MODE_MODEL_CPU,
        "emergency": OFFLOAD_MODE_MODEL_CPU,
        "modes": [OFFLOAD_MODE_MODEL_CPU],
    },
    # These are publisher recipe values, not a measured low-memory receipt.
    "lowVram": {
        "dtype": "bfloat16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_MODEL_CPU,
        "steps": 50,
        "width": 848,
        "height": 480,
        "numFrames": 121,
    },
    "modes": ["text_to_video", "image_to_video"],
    "modeDefaults": {
        "text_to_video": {"steps": 50, "guidanceScale": 6.0},
        "image_to_video": {"steps": 12, "guidanceScale": 1.0},
    },
    "modeRequirements": {
        "image_to_video": {
            "requiredImages": ["referenceImages"],
            "note": "Requires exactly one source image; width and height are source-derived unless both are set explicitly.",
        }
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(
            HUNYUAN_VIDEO_15_T2V_REPO,
            model_type="HunyuanVideo15ModularPipeline",
        ),
        require_catalog_revision(
            HUNYUAN_VIDEO_15_I2V_REPO,
            model_type="HunyuanVideo15ModularPipeline",
        ),
    ],
    "autoEligible": False,
    "templateEligible": False,
    "galleryEligible": False,
    "liveProof": False,
    "notes": [
        "T2V materializes text_encoder, denoise, and decode at 848x480; I2V preserves distinct VAE and SigLIP encoders before MeanFlow denoising.",
        "Both artifacts are immutable and safetensors-only; no quantization is selected or implied.",
        "Install and Run remain closed pending revision-bound license/AUP acknowledgement, territory and MAU eligibility enforcement, and downstream notice/disclosure UX.",
        "Resource floors, VAE-tiling behavior, real-weight execution, public promotion, Auto, templates, Gallery, and live proof remain unqualified.",
    ],
}

_HUNYUAN_VIDEO_15_STUDIO_EXECUTION_SPEC_DEFINITIONS = {
    "hunyuan-video15:modular-text-to-video:v1": {
        "modelType": "HunyuanVideo15ModularPipeline",
        "mode": "text_to_video",
        "profile": _hunyuan_video_15_profile("text_to_video", HUNYUAN_VIDEO_15_T2V_REPO, 50),
        "capability": deepcopy(_HUNYUAN_VIDEO_15_CAPABILITY),
        "roles": _HUNYUAN_VIDEO_15_T2V_GRAPH_ROLES,
        "edges": _HUNYUAN_VIDEO_15_T2V_GRAPH_EDGES,
        "bindings": _HUNYUAN_VIDEO_15_T2V_GRAPH_BINDINGS,
    },
    "hunyuan-video15:modular-image-to-video:v1": {
        "modelType": "HunyuanVideo15ModularPipeline",
        "mode": "image_to_video",
        "profile": _hunyuan_video_15_profile("image_to_video", HUNYUAN_VIDEO_15_I2V_REPO, 12),
        "capability": deepcopy(_HUNYUAN_VIDEO_15_CAPABILITY),
        "roles": _HUNYUAN_VIDEO_15_I2V_GRAPH_ROLES,
        "edges": _HUNYUAN_VIDEO_15_I2V_GRAPH_EDGES,
        "bindings": _HUNYUAN_VIDEO_15_I2V_GRAPH_BINDINGS,
    },
}

_MINIMAX_H3_MODULAR_PROFILE = {
    "id": "minimax-h3:official-modular-workflow",
    "model_type": "MiniMaxH3ModularPipeline",
    "modes": (
        "text_to_video_with_audio",
        "first_last_frame_to_video_with_audio",
        "reference_to_video_with_audio",
    ),
    "loader_module": "modules.ModularDiffusers",
    "loader_action": "ModelsLoader",
    "execution_path": "modular-diffusers",
    "pipeline_class": "MiniMaxH3ModularPipeline",
    "default_repo": MINIMAX_H3_REPO,
    "fallback_repo": None,
    # No quantization or offload recipe is claimed by the reviewed receipt.
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": (OFFLOAD_MODE_NONE,),
    "retry_offload_modes": (),
    "max_low_memory_side": None,
    "max_low_memory_steps": None,
    "live_proof": False,
    "compatible_repos": (),
}

_MINIMAX_H3_MODULAR_CAPABILITY = {
    "modelType": "MiniMaxH3ModularPipeline",
    "label": "MiniMax H3 (Modular Diffusers)",
    "displayName": "MiniMax-H3",
    "family": "MiniMax H3",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": MINIMAX_H3_REPO,
    # The reviewed receipt proves immutable weight partitions but not the
    # complete selective metadata/tokenizer closure. Keep app download
    # exposure closed instead of falling back to the 354 GB full repository.
    "downloadFiles": [],
    "artifactLabel": "Pinned safetensors repository with workflow-selected transformer partition",
    "license": "MiniMax-H3 Community License Agreement",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1344, "height": 768, "aspectRatio": "16:9"},
    "recommendedSteps": 50,
    "recommendedGuidance": 1.0,
    "guidanceLabel": "Guidance-distilled; no guidance control",
    "supportsNegativePrompt": False,
    "supportsImageInput": True,
    "supportsVideoInput": True,
    "supportsMask": False,
    "supportsMultiImage": True,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": False,
    "supportsAudioInput": True,
    "outputKind": "video",
    "outputMedia": ["video", "audio"],
    "recommendedFrames": 124,
    "recommendedFps": 24,
    "offloadSupport": {
        "default": OFFLOAD_MODE_NONE,
        "lowVram": OFFLOAD_MODE_NONE,
        "emergency": OFFLOAD_MODE_NONE,
        "modes": [OFFLOAD_MODE_NONE],
    },
    "lowVram": None,
    "modes": list(_MINIMAX_H3_MODULAR_PROFILE["modes"]),
    "modeRequirements": {
        "first_last_frame_to_video_with_audio": {
            "note": "Requires at least one decoded first or last keyframe; either or both are valid.",
        },
        "reference_to_video_with_audio": {
            "note": (
                "Requires an ordered list of official MiniMax H3 image/video/audio reference dataclasses. "
                "Audio-only lists are invalid; order is semantic."
            ),
        },
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(MINIMAX_H3_REPO, model_type="MiniMaxH3ModularPipeline")
    ],
    "autoEligible": False,
    "templateEligible": False,
    "galleryEligible": False,
    "liveProof": False,
    "notes": [
        "The loader seals t2va/fl2va to transformer and ref2va to transformer_ref; it never loads both partitions for one Cluster.",
        "Creator defaults are 1344x768, 124 frames at fixed 24 FPS, 50 steps, and seed 0; no negative prompt or guidance input is invented.",
        "Install and Run remain closed pending license/territory eligibility, an exact selective metadata closure, optional Transformers runtime, and measured 4-accelerator qualification.",
        "The current estimate-only floor is 160 GiB selective disk, 256 GiB system RAM, 192 GiB aggregate accelerator memory, and four accelerators.",
        "Public execution, Auto, templates, Gallery, and live proof remain disabled.",
    ],
}

_MINIMAX_H3_STUDIO_EXECUTION_SPEC_DEFINITIONS = {
    "minimax-h3:modular-text-to-video-with-audio:v1": {
        "modelType": "MiniMaxH3ModularPipeline",
        "mode": "text_to_video_with_audio",
        "profile": deepcopy(_MINIMAX_H3_MODULAR_PROFILE),
        "capability": deepcopy(_MINIMAX_H3_MODULAR_CAPABILITY),
        "roles": _MINIMAX_H3_T2VA_GRAPH_ROLES,
        "edges": _MINIMAX_H3_T2VA_GRAPH_EDGES,
        "bindings": _MINIMAX_H3_T2VA_GRAPH_BINDINGS,
    },
    "minimax-h3:modular-first-last-frame-to-video-with-audio:v1": {
        "modelType": "MiniMaxH3ModularPipeline",
        "mode": "first_last_frame_to_video_with_audio",
        "profile": deepcopy(_MINIMAX_H3_MODULAR_PROFILE),
        "capability": deepcopy(_MINIMAX_H3_MODULAR_CAPABILITY),
        "roles": _MINIMAX_H3_CONDITIONED_GRAPH_ROLES,
        "edges": _MINIMAX_H3_CONDITIONED_GRAPH_EDGES,
        "bindings": _MINIMAX_H3_FL2VA_GRAPH_BINDINGS,
    },
    "minimax-h3:modular-reference-to-video-with-audio:v1": {
        "modelType": "MiniMaxH3ModularPipeline",
        "mode": "reference_to_video_with_audio",
        "profile": deepcopy(_MINIMAX_H3_MODULAR_PROFILE),
        "capability": deepcopy(_MINIMAX_H3_MODULAR_CAPABILITY),
        "roles": _MINIMAX_H3_CONDITIONED_GRAPH_ROLES,
        "edges": _MINIMAX_H3_CONDITIONED_GRAPH_EDGES,
        "bindings": _MINIMAX_H3_REF2VA_GRAPH_BINDINGS,
    },
}

_HELIOS_MODEL_VARIANTS = (
    (
        "HeliosModularPipeline",
        "helios-base",
        "Helios Base",
        HELIOS_BASE_REPO,
        50,
        5.0,
    ),
    (
        "HeliosPyramidModularPipeline",
        "helios-pyramid",
        "Helios Pyramid",
        HELIOS_PYRAMID_REPO,
        10,
        5.0,
    ),
    (
        "HeliosPyramidDistilledModularPipeline",
        "helios-pyramid-distilled",
        "Helios Pyramid Distilled",
        HELIOS_DISTILLED_REPO,
        2,
        1.0,
    ),
)


def _helios_modular_profile(model_type, slug, repository, steps):
    return {
        "id": f"{slug}:official-modular-workflow",
        "model_type": model_type,
        "modes": ("text_to_video", "image_to_video", "video_to_video"),
        "loader_module": "modules.ModularDiffusers",
        "loader_action": "ModelsLoader",
        "execution_path": "modular-diffusers",
        "pipeline_class": model_type,
        "default_repo": repository,
        "fallback_repo": None,
        "quantizable_components": ("transformer", "text_encoder"),
        "default_quantized_components": (),
        "supported_offload_modes": (
            OFFLOAD_MODE_NONE,
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        ),
        "retry_offload_modes": (OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
        "max_low_memory_side": 640,
        "max_low_memory_steps": steps,
        "live_proof": False,
        "compatible_repos": (),
    }


def _helios_modular_capability(model_type, label, repository, steps, guidance):
    return {
        "modelType": model_type,
        "label": f"{label} (Modular Diffusers)",
        "displayName": label,
        "family": "Helios",
        "supportTier": "supported",
        "qualificationStatus": "graph-qualified-execution-pending",
        "qualifiedModes": [],
        "defaultRepo": repository,
        "downloadFiles": HELIOS_DIFFUSERS_FILES,
        "artifactLabel": "Exact reviewed 24-file Modular Diffusers component snapshot",
        "license": "Apache-2.0",
        "defaultDtype": "bfloat16",
        "defaultSize": {"width": 640, "height": 384, "aspectRatio": "5:3"},
        "recommendedSteps": steps,
        "recommendedGuidance": guidance,
        "guidanceLabel": "Official Helios guidance",
        "supportsImageInput": True,
        "supportsVideoInput": True,
        "supportsMask": False,
        "supportsMultiImage": False,
        "supportsControlImage": False,
        "supportsLayers": False,
        "supportsLora": True,
        "supportsAudioInput": False,
        "outputKind": "video",
        "recommendedFrames": 132,
        "recommendedFps": 24,
        "offloadSupport": {
            "default": OFFLOAD_MODE_GROUP_CPU,
            "lowVram": OFFLOAD_MODE_GROUP_DISK,
            "emergency": OFFLOAD_MODE_GROUP_DISK,
            "modes": [
                OFFLOAD_MODE_NONE,
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ],
        },
        "lowVram": {
            "dtype": "bfloat16",
            "autoOffload": True,
            "offloadMode": OFFLOAD_MODE_GROUP_DISK,
            "steps": steps,
            "width": 640,
            "height": 384,
            "numFrames": 132,
        },
        "modes": ["text_to_video", "image_to_video", "video_to_video"],
        "modeRequirements": {
            "image_to_video": {
                "requiredImages": ["referenceImages"],
                "note": "Requires exactly one source image.",
            },
            "video_to_video": {
                "requiredVideos": ["sourceVideo"],
                "note": "Requires a decoded source video long enough for the selected latent chunk size.",
            },
        },
        "executionStatus": "expert_only",
        "revisionCandidates": [require_catalog_revision(repository, model_type=model_type)],
        "autoEligible": False,
        "templateEligible": False,
        "galleryEligible": False,
        "liveProof": False,
        "notes": [
            "Runs the official Helios text encoder, conditional VAE encoder, denoise, and decode blocks.",
            "The exact component repository revision is propagated into every upstream component descriptor.",
            "Auto and Gallery remain disabled until remote heavy-hardware execution and output review are complete.",
        ],
    }


_HELIOS_WORKFLOW_GRAPHS = {
    "text_to_video": (
        _MODULAR_WHOLE_VIDEO_TEXT_GRAPH_ROLES,
        _MODULAR_WHOLE_VIDEO_TEXT_GRAPH_EDGES,
        _MODULAR_WHOLE_VIDEO_TEXT_GRAPH_BINDINGS,
    ),
    "image_to_video": (
        _MODULAR_WHOLE_VIDEO_IMAGE_GRAPH_ROLES,
        _MODULAR_WHOLE_VIDEO_IMAGE_GRAPH_EDGES,
        _MODULAR_WHOLE_VIDEO_IMAGE_GRAPH_BINDINGS,
    ),
    "video_to_video": (
        _MODULAR_WHOLE_VIDEO_EDIT_GRAPH_ROLES,
        _MODULAR_WHOLE_VIDEO_EDIT_GRAPH_EDGES,
        _MODULAR_WHOLE_VIDEO_EDIT_GRAPH_BINDINGS,
    ),
}
_HELIOS_STUDIO_EXECUTION_SPEC_DEFINITIONS = {
    f"{slug}:modular-{mode.replace('_', '-')}:v1": {
        "modelType": model_type,
        "mode": mode,
        "profile": _helios_modular_profile(model_type, slug, repository, steps),
        "capability": _helios_modular_capability(model_type, label, repository, steps, guidance),
        "roles": graph[0],
        "edges": graph[1],
        "bindings": graph[2],
    }
    for model_type, slug, label, repository, steps, guidance in _HELIOS_MODEL_VARIANTS
    for mode, graph in _HELIOS_WORKFLOW_GRAPHS.items()
}

_WAN_ANIMATE_2_MODEL_VARIANTS = (
    (
        "WanAnimate2ModularPipeline",
        "wan-animate-2",
        "Wan Animate 2",
        WAN_ANIMATE_2_REPO,
        40,
        3.0,
    ),
    (
        "WanAnimate2DistilledModularPipeline",
        "wan-animate-2-distilled",
        "Wan Animate 2 Distilled",
        WAN_ANIMATE_2_DISTILLED_REPO,
        10,
        1.0,
    ),
)


def _wan_animate_2_profile(model_type, slug, repository, steps):
    return {
        "id": f"{slug}:official-modular-workflow",
        "model_type": model_type,
        "modes": ("character_animate",),
        "loader_module": "modules.ModularDiffusers",
        "loader_action": "ModelsLoader",
        "execution_path": "modular-diffusers",
        "pipeline_class": model_type,
        "default_repo": repository,
        "fallback_repo": None,
        "quantizable_components": ("transformer", "text_encoder"),
        "default_quantized_components": (),
        "supported_offload_modes": (
            OFFLOAD_MODE_NONE,
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        ),
        "retry_offload_modes": (OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
        "max_low_memory_side": 640,
        "max_low_memory_steps": steps,
        "live_proof": False,
        "compatible_repos": (),
    }


def _wan_animate_2_capability(model_type, label, repository, steps, guidance):
    return {
        "modelType": model_type,
        "label": f"{label} (Modular Diffusers)",
        "displayName": label,
        "family": "Wan Animate 2",
        "supportTier": "supported",
        "qualificationStatus": "graph-qualified-execution-pending",
        "qualifiedModes": [],
        "defaultRepo": repository,
        "downloadFiles": WAN_ANIMATE_2_DIFFUSERS_FILES,
        "artifactLabel": "Exact reviewed 30-file Modular Diffusers component snapshot",
        "license": "Apache-2.0",
        "defaultDtype": "bfloat16",
        "defaultSize": {"width": 640, "height": 800, "aspectRatio": "4:5"},
        "recommendedSteps": steps,
        "recommendedGuidance": guidance,
        "guidanceLabel": "Official Wan Animate 2 guidance",
        "supportsImageInput": True,
        "supportsVideoInput": True,
        "supportsMask": False,
        "supportsMultiImage": False,
        "supportsControlImage": False,
        "supportsLayers": False,
        "supportsLora": True,
        "supportsAudioInput": False,
        "outputKind": "video",
        "recommendedFrames": 81,
        "recommendedFps": 24,
        "offloadSupport": {
            "default": OFFLOAD_MODE_GROUP_CPU,
            "lowVram": OFFLOAD_MODE_GROUP_DISK,
            "emergency": OFFLOAD_MODE_GROUP_DISK,
            "modes": [
                OFFLOAD_MODE_NONE,
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ],
        },
        "lowVram": {
            "dtype": "bfloat16",
            "autoOffload": True,
            "offloadMode": OFFLOAD_MODE_GROUP_DISK,
            "steps": steps,
            "width": 640,
            "height": 800,
            "numFrames": 81,
        },
        "modes": ["character_animate"],
        "modeRequirements": {
            "character_animate": {
                "requiredImages": ["referenceImages"],
                "requiredVideos": ["poseVideo"],
                "note": "Requires one reference-character image and one driving video; source FPS is preserved.",
            }
        },
        "executionStatus": "expert_only",
        "revisionCandidates": [require_catalog_revision(repository, model_type=model_type)],
        "autoEligible": False,
        "templateEligible": False,
        "galleryEligible": False,
        "liveProof": False,
        "notes": [
            "Runs the official text, reference-image CLIP, driving-video CLIP, reference VAE, segment denoise, and assembly blocks.",
            "Every same-repository component descriptor is normalized to the exact reviewed top-level commit.",
            "The Distilled definition fixes guidance to 1 and uses the official 10-step composed override.",
            "Auto and Gallery remain disabled until compiled Flex Attention remote execution and output review complete.",
        ],
    }


_WAN_ANIMATE_2_STUDIO_EXECUTION_SPEC_DEFINITIONS = {
    f"{slug}:modular-character-animate:v1": {
        "modelType": model_type,
        "mode": "character_animate",
        "profile": _wan_animate_2_profile(model_type, slug, repository, steps),
        "capability": _wan_animate_2_capability(model_type, label, repository, steps, guidance),
        "roles": _WAN_ANIMATE_2_GRAPH_ROLES,
        "edges": _WAN_ANIMATE_2_GRAPH_EDGES,
        "bindings": _WAN_ANIMATE_2_GRAPH_BINDINGS,
    }
    for model_type, slug, label, repository, steps, guidance in _WAN_ANIMATE_2_MODEL_VARIANTS
}


_COSMOS3_NANO_MODULAR_PROFILE = {
    "id": "cosmos3-nano:official-modular-workflow",
    "model_type": "Cosmos3OmniModularPipeline",
    "modes": _COSMOS3_NANO_STRUCTURAL_MODES,
    "loader_module": "modules.ModularDiffusers",
    "loader_action": "ModelsLoader",
    "execution_path": "modular-diffusers",
    "pipeline_class": "Cosmos3OmniModularPipeline",
    "default_repo": COSMOS3_NANO_REPO,
    "fallback_repo": None,
    # No quantization or offload recipe is claimed until the exact Nano
    # snapshot and mandatory guardrail have completed a measured run.
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": (OFFLOAD_MODE_NONE,),
    "retry_offload_modes": (),
    "max_low_memory_side": None,
    "max_low_memory_steps": None,
    "live_proof": False,
    "compatible_repos": (),
}

_COSMOS3_NANO_MODULAR_CAPABILITY = {
    "modelType": "Cosmos3OmniModularPipeline",
    "label": "Cosmos 3 Nano (Modular Diffusers)",
    "displayName": "Cosmos3-Nano",
    "family": "Cosmos 3",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": COSMOS3_NANO_REPO,
    "downloadFiles": COSMOS3_NANO_DIFFUSERS_FILES,
    "artifactLabel": "Exact reviewed 24-file Cosmos 3 Nano Modular Diffusers snapshot",
    "license": "OpenMDW-1.1",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1280, "height": 720, "aspectRatio": "16:9"},
    "recommendedSteps": 35,
    "recommendedGuidance": 6.0,
    "guidanceLabel": "Official Cosmos 3 Nano guidance",
    "supportsImageInput": True,
    "supportsVideoInput": True,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": False,
    "supportsAudioInput": False,
    "outputKind": "video",
    "modeOutputKinds": {
        "text_to_image": "image",
        "text_to_video": "video",
        "image_to_video": "video",
        "video_to_video": "video",
        "text_to_video_with_audio": "video",
        "image_to_video_with_audio": "video",
        "video_to_video_with_audio": "video",
    },
    "recommendedFrames": 189,
    "recommendedFps": 24,
    "offloadSupport": {
        "default": OFFLOAD_MODE_NONE,
        "lowVram": OFFLOAD_MODE_NONE,
        "emergency": OFFLOAD_MODE_NONE,
        "modes": [OFFLOAD_MODE_NONE],
    },
    "lowVram": None,
    "modes": list(_COSMOS3_NANO_STRUCTURAL_MODES),
    "modeRequirements": {
        "text_to_image": {
            "note": "The official text-to-image workflow seals num_frames to exactly 1.",
        },
        "text_to_video": {
            "note": "The official Nano model-card recipe uses 189 frames at 24 FPS.",
        },
        "image_to_video": {
            "requiredImages": ["referenceImages"],
            "note": "Requires exactly one source image for the official Cosmos 3 VAE conditioning block.",
        },
        "video_to_video": {
            "requiredVideos": ["sourceVideo"],
            "note": "Requires one decoded source video for the official Cosmos 3 VAE conditioning block.",
        },
        "text_to_video_with_audio": {
            "note": "The official sound workflow produces synchronized video and audio and is exported as one MP4.",
        },
        "image_to_video_with_audio": {
            "requiredImages": ["referenceImages"],
            "note": "Requires exactly one source image and exports synchronized generated video and audio.",
        },
        "video_to_video_with_audio": {
            "requiredVideos": ["sourceVideo"],
            "note": "Requires one decoded source video and exports synchronized generated video and audio.",
        },
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(COSMOS3_NANO_REPO, model_type="Cosmos3OmniModularPipeline")
    ],
    "autoEligible": False,
    "templateEligible": False,
    "galleryEligible": False,
    "liveProof": False,
    "notes": [
        "Runs the official Cosmos 3 text encoder, denoise, safety-checked decode, and after-decode blocks.",
        "The exact public Nano snapshot is graph-qualified only; no quantization or offload recipe is claimed.",
        "Execution remains closed until the mandatory gated Cosmos Guardrail revision is acknowledged, installed, and qualified.",
        "Auto, templates, Gallery, and public execution remain disabled until resource and output evidence is approved.",
    ],
}

_COSMOS3_NANO_WORKFLOW_GRAPHS = {
    "text_to_image": (
        "cosmos3-nano:modular-text-to-image:v1",
        _COSMOS3_OMNI_TEXT_TO_IMAGE_GRAPH_ROLES,
        _COSMOS3_OMNI_TEXT_TO_IMAGE_GRAPH_EDGES,
        _COSMOS3_OMNI_TEXT_TO_IMAGE_GRAPH_BINDINGS,
    ),
    "text_to_video": (
        "cosmos3-nano:modular-text-to-video:v1",
        _COSMOS3_OMNI_TEXT_TO_VIDEO_GRAPH_ROLES,
        _COSMOS3_OMNI_TEXT_TO_VIDEO_GRAPH_EDGES,
        _COSMOS3_OMNI_TEXT_TO_VIDEO_GRAPH_BINDINGS,
    ),
    "image_to_video": (
        "cosmos3-nano:modular-image-to-video:v1",
        _COSMOS3_OMNI_IMAGE_TO_VIDEO_GRAPH_ROLES,
        _COSMOS3_OMNI_IMAGE_TO_VIDEO_GRAPH_EDGES,
        _COSMOS3_OMNI_IMAGE_TO_VIDEO_GRAPH_BINDINGS,
    ),
    "video_to_video": (
        "cosmos3-nano:modular-video-to-video:v1",
        _COSMOS3_OMNI_VIDEO_TO_VIDEO_GRAPH_ROLES,
        _COSMOS3_OMNI_VIDEO_TO_VIDEO_GRAPH_EDGES,
        _COSMOS3_OMNI_VIDEO_TO_VIDEO_GRAPH_BINDINGS,
    ),
    "text_to_video_with_audio": (
        "cosmos3-nano:modular-text-to-video-with-audio:v1",
        _COSMOS3_OMNI_TEXT_TO_VIDEO_WITH_AUDIO_GRAPH_ROLES,
        _COSMOS3_OMNI_TEXT_TO_VIDEO_WITH_AUDIO_GRAPH_EDGES,
        _COSMOS3_OMNI_TEXT_TO_VIDEO_GRAPH_BINDINGS,
    ),
    "image_to_video_with_audio": (
        "cosmos3-nano:modular-image-to-video-with-audio:v1",
        _COSMOS3_OMNI_IMAGE_TO_VIDEO_WITH_AUDIO_GRAPH_ROLES,
        _COSMOS3_OMNI_IMAGE_TO_VIDEO_WITH_AUDIO_GRAPH_EDGES,
        _COSMOS3_OMNI_IMAGE_TO_VIDEO_GRAPH_BINDINGS,
    ),
    "video_to_video_with_audio": (
        "cosmos3-nano:modular-video-to-video-with-audio:v1",
        _COSMOS3_OMNI_VIDEO_TO_VIDEO_WITH_AUDIO_GRAPH_ROLES,
        _COSMOS3_OMNI_VIDEO_TO_VIDEO_WITH_AUDIO_GRAPH_EDGES,
        _COSMOS3_OMNI_VIDEO_TO_VIDEO_GRAPH_BINDINGS,
    ),
}

_COSMOS3_NANO_STUDIO_EXECUTION_SPEC_DEFINITIONS = {
    spec_id: {
        "modelType": "Cosmos3OmniModularPipeline",
        "mode": mode,
        "profile": deepcopy(_COSMOS3_NANO_MODULAR_PROFILE),
        "capability": deepcopy(_COSMOS3_NANO_MODULAR_CAPABILITY),
        "roles": roles,
        "edges": edges,
        "bindings": bindings,
        # The official workflow always runs ``after_decode``. Non-action
        # routes return ``action=None`` from that side-output block, so it is
        # an intentional terminal beside the user-facing media sink.
        "auxiliaryTerminalRoles": ("afterDecode",),
    }
    for mode, (spec_id, roles, edges, bindings) in _COSMOS3_NANO_WORKFLOW_GRAPHS.items()
}


def _cosmos3_distilled_profile(mode, repository):
    slug = "text-to-image" if mode == "text_to_image" else "image-to-video"
    return {
        "id": f"cosmos3-distilled-{slug}:official-modular-workflow",
        "model_type": "Cosmos3DistilledModularPipeline",
        "modes": (mode,),
        "loader_module": "modules.ModularDiffusers",
        "loader_action": "ModelsLoader",
        "execution_path": "modular-diffusers",
        "pipeline_class": "Cosmos3DistilledModularPipeline",
        "default_repo": repository,
        "fallback_repo": None,
        "quantizable_components": (),
        "default_quantized_components": (),
        # This preserves the reviewed publisher recipe; it is not a measured
        # resource qualification or permission to execute the 130 GB snapshot.
        "supported_offload_modes": (OFFLOAD_MODE_NONE,),
        "retry_offload_modes": (),
        "max_low_memory_side": None,
        "max_low_memory_steps": None,
        "live_proof": False,
        "compatible_repos": (),
    }


_COSMOS3_DISTILLED_MODULAR_CAPABILITY = {
    "modelType": "Cosmos3DistilledModularPipeline",
    "label": "Cosmos 3 Distilled (Modular Diffusers)",
    "displayName": "Cosmos3-Super 4-Step",
    "family": "Cosmos 3",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": COSMOS3_DISTILLED_T2I_REPO,
    "downloadFiles": COSMOS3_DISTILLED_T2I_DIFFUSERS_FILES,
    "artifactSelections": [
        {
            "modes": ["text_to_image"],
            "repo": COSMOS3_DISTILLED_T2I_REPO,
            "revision": require_catalog_revision(
                COSMOS3_DISTILLED_T2I_REPO,
                model_type="Cosmos3DistilledModularPipeline",
            ),
            "downloadFiles": COSMOS3_DISTILLED_T2I_DIFFUSERS_FILES,
            "label": "Exact reviewed Cosmos 3 Super text-to-image 4-step snapshot",
        },
        {
            "modes": ["image_to_video"],
            "repo": COSMOS3_DISTILLED_I2V_REPO,
            "revision": require_catalog_revision(
                COSMOS3_DISTILLED_I2V_REPO,
                model_type="Cosmos3DistilledModularPipeline",
            ),
            "downloadFiles": COSMOS3_DISTILLED_I2V_DIFFUSERS_FILES,
            "label": "Exact reviewed Cosmos 3 Super image-to-video 4-step snapshot",
        },
    ],
    "artifactLabel": "Exact reviewed route-specific Cosmos 3 Super 4-step snapshot",
    "license": "OpenMDW-1.1",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1280, "height": 720, "aspectRatio": "16:9"},
    "recommendedSteps": 4,
    "recommendedGuidance": 1.0,
    "guidanceLabel": "Fixed distilled guidance",
    "supportsImageInput": True,
    "supportsVideoInput": False,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": False,
    "supportsAudioInput": False,
    "outputKind": "video",
    "modeOutputKinds": {"text_to_image": "image", "image_to_video": "video"},
    "recommendedFrames": 189,
    "recommendedFps": 24,
    "offloadSupport": {
        "default": OFFLOAD_MODE_NONE,
        "lowVram": OFFLOAD_MODE_NONE,
        "emergency": OFFLOAD_MODE_NONE,
        "modes": [OFFLOAD_MODE_NONE],
    },
    "lowVram": None,
    "modes": list(_COSMOS3_DISTILLED_STRUCTURAL_MODES),
    "modeRequirements": {
        "text_to_image": {
            "note": "The exact upstream workflow seals num_frames to one and the distilled schedule to four steps.",
        },
        "image_to_video": {
            "requiredImages": ["referenceImages"],
            "note": "Requires exactly one source image and the route-specific image-to-video checkpoint.",
        },
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(
            COSMOS3_DISTILLED_T2I_REPO,
            model_type="Cosmos3DistilledModularPipeline",
        ),
        require_catalog_revision(
            COSMOS3_DISTILLED_I2V_REPO,
            model_type="Cosmos3DistilledModularPipeline",
        ),
    ],
    "autoEligible": False,
    "templateEligible": False,
    "galleryEligible": False,
    "liveProof": False,
    "notes": [
        "Runs the exact Distilled text encoder, optional image VAE encoder, fixed-schedule denoise, and decode blocks.",
        "Negative prompts are unsupported; inference steps and guidance are sealed to the official four-step schedule and 1.0.",
        "Execution remains closed until the mandatory gated Cosmos Guardrail, tensor-parallel resource recipe, and output are qualified.",
        "Auto, templates, Gallery, and public execution remain disabled.",
    ],
}

_COSMOS3_DISTILLED_STUDIO_EXECUTION_SPEC_DEFINITIONS = {
    "cosmos3-distilled:modular-text-to-image:v1": {
        "modelType": "Cosmos3DistilledModularPipeline",
        "mode": "text_to_image",
        "profile": _cosmos3_distilled_profile("text_to_image", COSMOS3_DISTILLED_T2I_REPO),
        "capability": deepcopy(_COSMOS3_DISTILLED_MODULAR_CAPABILITY),
        "roles": _COSMOS3_DISTILLED_T2I_GRAPH_ROLES,
        "edges": _COSMOS3_DISTILLED_T2I_GRAPH_EDGES,
        "bindings": _COSMOS3_DISTILLED_T2I_GRAPH_BINDINGS,
    },
    "cosmos3-distilled:modular-image-to-video:v1": {
        "modelType": "Cosmos3DistilledModularPipeline",
        "mode": "image_to_video",
        "profile": _cosmos3_distilled_profile("image_to_video", COSMOS3_DISTILLED_I2V_REPO),
        "capability": deepcopy(_COSMOS3_DISTILLED_MODULAR_CAPABILITY),
        "roles": _COSMOS3_DISTILLED_I2V_GRAPH_ROLES,
        "edges": _COSMOS3_DISTILLED_I2V_GRAPH_EDGES,
        "bindings": _COSMOS3_DISTILLED_I2V_GRAPH_BINDINGS,
    },
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
        "capability": {
            **_capability(
                "FluxSchnellPipeline",
                "FLUX.1 schnell",
                "FLUX.1-schnell",
                FLUX_SCHNELL_REPO,
                width=1024,
                steps=4,
                guidance=0.0,
                low_vram_mode=OFFLOAD_MODE_MODEL_CPU,
            ),
            "downloadFiles": FLUX_SCHNELL_DIFFUSERS_FILES,
        },
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
            "downloadFiles": FLUX_DEV_DIFFUSERS_FILES,
            "notes": ["Auto prefers the FP8 artifact on 16 GB CUDA when available."],
            "revisionCandidates": [require_catalog_revision(FLUX_DEV_REPO, model_type="FluxDevPipeline")],
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
                },
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
        "capability": {
            **_capability(
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
            "downloadFiles": FLUX_KREA_DIFFUSERS_FILES,
        },
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
            "downloadFiles": FLUX_CONTROL_DIFFUSERS_FILES,
            "supportsImageInput": True,
            "supportsMask": True,
            "supportsControlImage": True,
            "modes": ["control_image", "control_edit_image", "control_inpaint"],
            "modeRequirements": {
                "control_image": {
                    "requiredImages": ["controlImage"],
                    "note": "Requires one prepared depth control image.",
                },
                "control_edit_image": {
                    "requiredImages": ["referenceImages", "controlImage"],
                    "note": "Requires one source image and one prepared depth control image.",
                },
                "control_inpaint": {
                    "requiredImages": ["referenceImages", "maskImage", "controlImage"],
                    "note": "Requires one source image, one mask, and one prepared depth control image.",
                },
            },
        },
        "autoRequirements": {
            "supportedTasks": ["control_image", "control_edit_image", "control_inpaint"],
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
            "downloadFiles": FLUX_CONTROL_DIFFUSERS_FILES,
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
            "supportsMask": True,
            "supportsControlImage": True,
            "modes": ["control_image", "control_edit_image", "control_inpaint"],
            "modeRequirements": {
                "control_image": {
                    "requiredImages": ["controlImage"],
                    "note": "Requires one prepared Canny control image.",
                },
                "control_edit_image": {
                    "requiredImages": ["referenceImages", "controlImage"],
                    "note": "Requires one source image and one prepared Canny control image.",
                },
                "control_inpaint": {
                    "requiredImages": ["referenceImages", "maskImage", "controlImage"],
                    "note": "Requires one source image, one mask, and one prepared Canny control image.",
                },
            },
        },
        "autoRequirements": {
            "supportedTasks": ["control_image", "control_edit_image", "control_inpaint"],
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
    "flux-depth:control-edit-image:v1": {
        "modelType": "FluxDepthPipeline",
        "mode": "control_edit_image",
        "profile": _profile(
            "flux-depth:img2img-direct",
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
            mode="control_edit_image",
            pipeline_class="FluxControlImg2ImgPipeline",
        ),
        "roles": _CONTROL_EDIT_GRAPH_ROLES,
        "edges": _CONTROL_EDIT_GRAPH_EDGES,
        "bindings": _CONTROL_EDIT_GRAPH_BINDINGS,
    },
    "flux-depth:control-inpaint:v1": {
        "modelType": "FluxDepthPipeline",
        "mode": "control_inpaint",
        "profile": _profile(
            "flux-depth:inpaint-direct",
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
            mode="control_inpaint",
            pipeline_class="FluxControlInpaintPipeline",
        ),
        "roles": _CONTROL_INPAINT_GRAPH_ROLES,
        "edges": _CONTROL_INPAINT_GRAPH_EDGES,
        "bindings": _CONTROL_INPAINT_GRAPH_BINDINGS,
    },
    "flux-canny:control-edit-image:v1": {
        "modelType": "FluxCannyPipeline",
        "mode": "control_edit_image",
        "profile": _profile(
            "flux-canny:img2img-direct",
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
            mode="control_edit_image",
            pipeline_class="FluxControlImg2ImgPipeline",
        ),
        "roles": _CONTROL_EDIT_GRAPH_ROLES,
        "edges": _CONTROL_EDIT_GRAPH_EDGES,
        "bindings": _CONTROL_EDIT_GRAPH_BINDINGS,
    },
    "flux-canny:control-inpaint:v1": {
        "modelType": "FluxCannyPipeline",
        "mode": "control_inpaint",
        "profile": _profile(
            "flux-canny:inpaint-direct",
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
            mode="control_inpaint",
            pipeline_class="FluxControlInpaintPipeline",
        ),
        "roles": _CONTROL_INPAINT_GRAPH_ROLES,
        "edges": _CONTROL_INPAINT_GRAPH_EDGES,
        "bindings": _CONTROL_INPAINT_GRAPH_BINDINGS,
    },
    "flux-redux:edit-image:v1": {
        "modelType": "FluxReduxPipeline",
        "mode": "edit_image",
        "profile": _flux_redux_profile(),
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
            "downloadFiles": FLUX_REDUX_DIFFUSERS_FILES,
            "artifactCandidates": [FLUX_REDUX_REPO, FLUX_DEV_REPO],
            "supportsImageInput": True,
            "supportsMultiImage": True,
            "modes": ["edit_image", "multi_image_reference_edit"],
            "additionalRequirements": studio_model_requirements_for_pair("FluxReduxPipeline", "edit_image"),
            "modeRequirements": {
                "edit_image": {
                    "modelRequirements": studio_model_requirements_for_pair("FluxReduxPipeline", "edit_image"),
                    "requiredImages": ["referenceImages"],
                    "note": "Requires reference images plus the reviewed FLUX.1-dev base pipeline.",
                },
                "multi_image_reference_edit": {
                    "modelRequirements": studio_model_requirements_for_pair(
                        "FluxReduxPipeline", "multi_image_reference_edit"
                    ),
                    "requiredImages": ["referenceImages"],
                    "note": "Requires multiple reference images plus the reviewed FLUX.1-dev base pipeline.",
                },
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
            "downloadFiles": FLUX2_KLEIN_DIFFUSERS_FILES,
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
            "revisionCandidates": [require_catalog_revision(WAN_22_I2V_A14B_REPO)],
            "downloadFiles": WAN_22_I2V_A14B_DIFFUSERS_FILES,
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
                "The native BF16 A14B artifact requires disk-group offload on hosts that cannot safely retain both experts in system RAM.",
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
            "offloadRequirements": _WAN_22_A14B_EXPERT_RESOURCE_REQUIREMENTS,
            "requiredPackages": ["diffusers", "transformers", "accelerate", "torch"],
            "guardedReason": "Wan 2.2 I2V A14B uses the generic Diffusers video graph with a dual-transformer execution contract.",
        },
        "expertResourceRequirements": _WAN_22_A14B_EXPERT_RESOURCE_REQUIREMENTS,
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
            "revisionCandidates": [require_catalog_revision(WAN_22_TI2V_5B_REPO)],
            "downloadFiles": WAN_22_TI2V_5B_DIFFUSERS_FILES,
            "artifactLabel": "Diffusers repo",
            "defaultDtype": "bfloat16",
            "defaultSize": {"width": 1280, "height": 704, "aspectRatio": "16:9"},
            "recommendedSteps": 50,
            "recommendedGuidance": 5.0,
            "guidanceLabel": "Guidance",
            # Diffusers currently exposes this dense 5B checkpoint through
            # ``WanPipeline.__call__`` as text-to-video.  The upstream model
            # family name includes TI2V, but that must not be interpreted as
            # an admitted image input on the standard Diffusers route.
            "supportsImageInput": False,
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
        # The standard Cluster route seals the immutable artifact revision and
        # execution profile on the loader itself.  A generic direct workflow
        # must not silently fall back to a moving revision or a different
        # loader profile after save/refresh.
        "bindings": tuple(
            (role, param, "defaultRevision")
            if (role, param) == ("wanPipeline", "revision")
            else item
            for item in _VIDEO_GRAPH_BINDINGS
            for role, param, _source in (item,)
        )
        + (("wanPipeline", "execution_profile_id", "executionProfileId"),),
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
        "bindings": tuple(
            (role, param, "nativeMath") if (role, param) == ("diffusersRecipe", "attention_backend") else item
            for item in _GRAPH_BINDINGS
            for role, param, _source in (item,)
        ),
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
        "bindings": tuple(
            (role, param, "nativeMath") if (role, param) == ("diffusersRecipe", "attention_backend") else item
            for item in _EDIT_GRAPH_BINDINGS
            for role, param, _source in (item,)
        )
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
        "bindings": _EDIT_GRAPH_BINDINGS + (("diffusersImagePipeline", "revision", "defaultRevision"),),
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
        "bindings": _INPAINT_GRAPH_BINDINGS + (("diffusersImagePipeline", "revision", "defaultRevision"),),
    },
    "qwen-image-edit:edit-image:v1": {
        "modelType": "QwenImageEditModularPipeline",
        "mode": "edit_image",
        "profile": _MODULAR_QWEN_EDIT_PROFILE,
        "roles": _MODULAR_EDIT_GRAPH_ROLES,
        "edges": _MODULAR_EDIT_GRAPH_EDGES,
        "bindings": _MODULAR_EDIT_GRAPH_BINDINGS,
    },
    "qwen-image-edit:modular-inpainting:v1": {
        "modelType": "QwenImageEditModularPipeline",
        "mode": "modular_inpainting",
        "profile": _MODULAR_QWEN_EDIT_PROFILE,
        "roles": _MODULAR_SDXL_INPAINT_GRAPH_ROLES,
        "edges": _MODULAR_QWEN_EDIT_INPAINT_GRAPH_EDGES,
        "bindings": _MODULAR_QWEN_EDIT_INPAINT_GRAPH_BINDINGS,
        "autoRequirementKey": "QwenImageEditModularPipeline:modular_inpainting",
        "autoRequirements": _MODULAR_QWEN_EDIT_INPAINT_AUTO_REQUIREMENTS,
    },
    "flux-dev:modular-text-to-image:v1": {
        "modelType": "FluxModularPipeline",
        "mode": "text_to_image",
        "profile": _MODULAR_FLUX_PROFILE,
        "roles": _MODULAR_TEXT_TO_IMAGE_GRAPH_ROLES,
        "edges": _MODULAR_FLUX_TEXT_TO_IMAGE_GRAPH_EDGES + _MODULAR_FLUX_DECODE_GEOMETRY_EDGES,
        "bindings": _MODULAR_FLUX_TEXT_TO_IMAGE_GRAPH_BINDINGS,
        "capability": _MODULAR_FLUX_CAPABILITY,
        "autoRequirements": _MODULAR_FLUX_AUTO_REQUIREMENTS,
    },
    "flux-dev:modular-image-to-image:v1": {
        "modelType": "FluxModularPipeline",
        "mode": "image_to_image",
        "profile": _MODULAR_FLUX_PROFILE,
        "roles": _MODULAR_SDXL_EDIT_GRAPH_ROLES,
        "edges": _MODULAR_FLUX_IMAGE_TO_IMAGE_GRAPH_EDGES + _MODULAR_FLUX_DECODE_GEOMETRY_EDGES,
        "bindings": tuple(
            binding
            for binding in _MODULAR_SDXL_EDIT_GRAPH_BINDINGS
            if binding != ("prompt", "negative_prompt", "negativePrompt")
        ) + _MODULAR_FLUX_IMAGE_ENCODE_GEOMETRY_BINDINGS,
    },
    "flux-kontext:modular-text-to-image:v1": {
        "modelType": "FluxKontextModularPipeline",
        "mode": "text_to_image",
        "profile": _MODULAR_FLUX_KONTEXT_PROFILE,
        "roles": _MODULAR_TEXT_TO_IMAGE_GRAPH_ROLES,
        "edges": _MODULAR_FLUX_TEXT_TO_IMAGE_GRAPH_EDGES + _MODULAR_FLUX_DECODE_GEOMETRY_EDGES,
        "bindings": _MODULAR_FLUX_TEXT_TO_IMAGE_GRAPH_BINDINGS,
        "capability": _MODULAR_FLUX_KONTEXT_CAPABILITY,
        "autoRequirements": _MODULAR_FLUX_KONTEXT_AUTO_REQUIREMENTS,
    },
    "flux-kontext:modular-edit-image:v1": {
        "modelType": "FluxKontextModularPipeline",
        "mode": "edit_image",
        "profile": _MODULAR_FLUX_KONTEXT_PROFILE,
        "roles": _MODULAR_SDXL_EDIT_GRAPH_ROLES,
        "edges": _MODULAR_FLUX_IMAGE_TO_IMAGE_GRAPH_EDGES + _MODULAR_FLUX_DECODE_GEOMETRY_EDGES,
        "bindings": tuple(
            binding
            for binding in _MODULAR_SDXL_EDIT_GRAPH_BINDINGS
            if binding
            not in {
                ("prompt", "negative_prompt", "negativePrompt"),
                ("denoise", "strength", "strength"),
            }
        ),
    },
    "flux2-klein:modular-text-to-image:v1": {
        "modelType": "Flux2KleinModularPipeline",
        "mode": "text_to_image",
        "profile": _MODULAR_FLUX2_KLEIN_PROFILE,
        "roles": _MODULAR_TEXT_TO_IMAGE_GRAPH_ROLES,
        "edges": _MODULAR_FLUX_TEXT_TO_IMAGE_GRAPH_EDGES,
        "bindings": _MODULAR_FLUX_TEXT_TO_IMAGE_GRAPH_BINDINGS,
        "capability": _MODULAR_FLUX2_KLEIN_CAPABILITY,
        "autoRequirements": _MODULAR_FLUX2_KLEIN_AUTO_REQUIREMENTS,
    },
    "flux2-klein:modular-edit-image:v1": {
        "modelType": "Flux2KleinModularPipeline",
        "mode": "edit_image",
        "profile": _MODULAR_FLUX2_KLEIN_PROFILE,
        "roles": _MODULAR_SDXL_EDIT_GRAPH_ROLES,
        "edges": _MODULAR_FLUX_IMAGE_TO_IMAGE_GRAPH_EDGES,
        "bindings": tuple(
            binding
            for binding in _MODULAR_SDXL_EDIT_GRAPH_BINDINGS
            if binding
            not in {
                ("prompt", "negative_prompt", "negativePrompt"),
                ("denoise", "strength", "strength"),
            }
        ),
    },
    "flux2-klein-base:modular-text-to-image:v1": {
        "modelType": "Flux2KleinBaseModularPipeline",
        "mode": "text_to_image",
        "profile": _MODULAR_FLUX2_KLEIN_BASE_PROFILE,
        "roles": _MODULAR_TEXT_TO_IMAGE_GRAPH_ROLES,
        "edges": _MODULAR_FLUX_TEXT_TO_IMAGE_GRAPH_EDGES,
        "bindings": _MODULAR_FLUX_TEXT_TO_IMAGE_GRAPH_BINDINGS,
        "capability": _MODULAR_FLUX2_KLEIN_BASE_CAPABILITY,
        "autoRequirements": _MODULAR_FLUX2_KLEIN_BASE_AUTO_REQUIREMENTS,
    },
    "flux2-klein-base:modular-edit-image:v1": {
        "modelType": "Flux2KleinBaseModularPipeline",
        "mode": "edit_image",
        "profile": _MODULAR_FLUX2_KLEIN_BASE_PROFILE,
        "roles": _MODULAR_SDXL_EDIT_GRAPH_ROLES,
        "edges": _MODULAR_FLUX_IMAGE_TO_IMAGE_GRAPH_EDGES,
        "bindings": tuple(
            binding
            for binding in _MODULAR_SDXL_EDIT_GRAPH_BINDINGS
            if binding
            not in {
                ("prompt", "negative_prompt", "negativePrompt"),
                ("denoise", "strength", "strength"),
            }
        ),
    },
    "z-image:modular-text-to-image:v1": {
        "modelType": "ZImageModularPipeline",
        "mode": "modular_text_to_image",
        "profile": _MODULAR_Z_IMAGE_PROFILE,
        "roles": _MODULAR_TEXT_TO_IMAGE_GRAPH_ROLES,
        "edges": _MODULAR_Z_IMAGE_TEXT_TO_IMAGE_GRAPH_EDGES,
        "bindings": _MODULAR_FLUX_TEXT_TO_IMAGE_GRAPH_BINDINGS,
        "autoRequirementKey": "ZImageModularPipeline:modular_text_to_image",
        "autoRequirements": _MODULAR_Z_IMAGE_AUTO_REQUIREMENTS,
    },
    "z-image:modular-image-to-image:v1": {
        "modelType": "ZImageModularPipeline",
        "mode": "modular_image_to_image",
        "profile": _MODULAR_Z_IMAGE_PROFILE,
        "roles": _MODULAR_SDXL_EDIT_GRAPH_ROLES,
        "edges": _MODULAR_Z_IMAGE_TO_IMAGE_GRAPH_EDGES,
        "bindings": tuple(
            binding
            for binding in _MODULAR_SDXL_EDIT_GRAPH_BINDINGS
            if binding != ("prompt", "negative_prompt", "negativePrompt")
        ) + _MODULAR_FLUX_IMAGE_ENCODE_GEOMETRY_BINDINGS,
        "autoRequirementKey": "ZImageModularPipeline:modular_image_to_image",
        "autoRequirements": _MODULAR_Z_IMAGE_AUTO_REQUIREMENTS,
    },
    "sdxl-base:modular-text-to-image:v1": {
        "modelType": "StableDiffusionXLModularPipeline",
        "mode": "text_to_image",
        "profile": _MODULAR_SDXL_PROFILE,
        "roles": _MODULAR_TEXT_TO_IMAGE_GRAPH_ROLES,
        "edges": _MODULAR_TEXT_TO_IMAGE_GRAPH_EDGES,
        "bindings": _MODULAR_TEXT_TO_IMAGE_GRAPH_BINDINGS,
        "capability": {
            "modelType": "StableDiffusionXLModularPipeline",
            "label": "Stable Diffusion XL 1.0 (Modular Diffusers)",
            "displayName": "stable-diffusion-xl-base-1.0",
            "family": "Stable Diffusion XL",
            "supportTier": "supported",
            "qualificationStatus": "graph-qualified-execution-pending",
            "qualifiedModes": [],
            "defaultRepo": SDXL_BASE_REPO,
            "downloadFiles": SDXL_BASE_FP16_DIFFUSERS_FILES,
            "artifactLabel": "Pinned Diffusers fp16 safetensors repository",
            "defaultDtype": "float16",
            "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
            "recommendedSteps": 30,
            "recommendedGuidance": 5.0,
            "guidanceLabel": "Guidance",
            "supportsNegativePrompt": True,
            "supportsImageInput": True,
            "supportsMask": True,
            "supportsMultiImage": False,
            "supportsControlImage": True,
            "supportsLayers": False,
            "supportsLora": False,
            "offloadSupport": {
                "default": OFFLOAD_MODE_MODEL_CPU,
                "lowVram": OFFLOAD_MODE_MODEL_CPU,
                "emergency": OFFLOAD_MODE_GROUP_DISK,
                "modes": list(_MODULAR_SDXL_PROFILE["supported_offload_modes"]),
            },
            "lowVram": {
                "dtype": "float16",
                "autoOffload": True,
                "offloadMode": OFFLOAD_MODE_MODEL_CPU,
                "steps": 24,
                "width": 768,
                "height": 768,
            },
            "modes": list(_MODULAR_SDXL_PROFILE["modes"]),
            "modeRequirements": {
                "edit_image": {"requiredImages": ["referenceImages"]},
                "inpaint": {"requiredImages": ["referenceImages", "maskImage"]},
                "control_image": {
                    "requiredImages": ["controlImage"],
                    "modelRequirements": studio_model_requirements_for_pair(
                        "StableDiffusionXLModularPipeline", "control_image"
                    ),
                    "note": "Requires one prepared Canny control image.",
                },
                "control_edit_image": {
                    "requiredImages": ["referenceImages", "controlImage"],
                    "modelRequirements": studio_model_requirements_for_pair(
                        "StableDiffusionXLModularPipeline", "control_edit_image"
                    ),
                    "note": "Requires one source image and one prepared Canny control image.",
                },
                "control_inpaint": {
                    "requiredImages": ["referenceImages", "maskImage", "controlImage"],
                    "modelRequirements": studio_model_requirements_for_pair(
                        "StableDiffusionXLModularPipeline", "control_inpaint"
                    ),
                    "note": "Requires one source image, one mask image, and one prepared Canny control image.",
                },
            },
            "revisionCandidates": [require_catalog_revision(SDXL_BASE_REPO)],
            "executionStatus": "expert_only",
            "autoEligible": True,
            "templateEligible": True,
            "galleryEligible": False,
            "liveProof": False,
            "notes": [
                "This capability represents the admitted Modular Diffusers text-to-image Cluster route, not the separate standard Diffusers pipeline route.",
                "Auto plans resources per exact Cluster instance; Gallery publication remains disabled until visible-output qualification is complete.",
            ],
        },
        "autoRequirements": {
            "supportedTasks": list(_MODULAR_SDXL_PROFILE["modes"]),
            "defaultRepo": SDXL_BASE_REPO,
            "qualityDefaults": {
                "width": 1024,
                "height": 1024,
                "steps": 30,
                "guidanceScale": 5,
                "maxSequenceLength": 77,
            },
            "minimum": {
                "accelerator": "gpu_or_cpu",
                "vramBytes": 0,
                "systemRamBytes": 16 * _GIB,
                "diskFreeBytes": 15 * _GIB,
            },
            "recommended": {
                "accelerator": "gpu",
                "vramBytes": 24 * _GIB,
                "systemRamBytes": 32 * _GIB,
                "diskFreeBytes": 25 * _GIB,
            },
            "fullResidency": {
                "accelerator": "gpu",
                "vramBytes": 32 * _GIB,
                "systemRamBytes": 32 * _GIB,
            },
            "supportedOffloadModes": [
                OFFLOAD_MODE_NONE,
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ],
            "requiredPackages": ["diffusers", "transformers", "accelerate", "torch"],
            "guardedReason": (
                "SDXL Modular Diffusers uses one independently planned Cluster instance; auxiliary ControlNet and "
                "IP-Adapter dependencies are joined from the exact selected workflow without sharing mutable state "
                "between Cluster Nodes."
            ),
        },
    },
    "sdxl-base:modular-image-to-image:v1": {
        "modelType": "StableDiffusionXLModularPipeline",
        "mode": "edit_image",
        "profile": _MODULAR_SDXL_PROFILE,
        "roles": _MODULAR_SDXL_EDIT_GRAPH_ROLES,
        "edges": _MODULAR_SDXL_EDIT_GRAPH_EDGES,
        "bindings": _MODULAR_SDXL_EDIT_GRAPH_BINDINGS,
    },
    "sdxl-base:modular-inpainting:v1": {
        "modelType": "StableDiffusionXLModularPipeline",
        "mode": "inpaint",
        "profile": _MODULAR_SDXL_PROFILE,
        "roles": _MODULAR_SDXL_INPAINT_GRAPH_ROLES,
        "edges": _MODULAR_SDXL_INPAINT_GRAPH_EDGES,
        "bindings": _MODULAR_SDXL_INPAINT_GRAPH_BINDINGS,
    },
    "sdxl-base:modular-controlnet-text-to-image:v1": {
        "modelType": "StableDiffusionXLModularPipeline",
        "mode": "control_image",
        "profile": _MODULAR_SDXL_PROFILE,
        "roles": _MODULAR_SDXL_CONTROL_GRAPH_ROLES,
        "edges": _MODULAR_SDXL_CONTROL_GRAPH_EDGES,
        "bindings": _MODULAR_SDXL_CONTROL_GRAPH_BINDINGS,
    },
    "sdxl-base:modular-controlnet-image-to-image:v1": {
        "modelType": "StableDiffusionXLModularPipeline",
        "mode": "control_edit_image",
        "profile": _MODULAR_SDXL_PROFILE,
        "roles": _MODULAR_SDXL_CONTROL_EDIT_GRAPH_ROLES,
        "edges": _MODULAR_SDXL_CONTROL_EDIT_GRAPH_EDGES,
        "bindings": _MODULAR_SDXL_CONTROL_EDIT_GRAPH_BINDINGS,
    },
    "sdxl-base:modular-controlnet-inpainting:v1": {
        "modelType": "StableDiffusionXLModularPipeline",
        "mode": "control_inpaint",
        "profile": _MODULAR_SDXL_PROFILE,
        "roles": _MODULAR_SDXL_CONTROL_INPAINT_GRAPH_ROLES,
        "edges": _MODULAR_SDXL_CONTROL_INPAINT_GRAPH_EDGES,
        "bindings": _MODULAR_SDXL_CONTROL_INPAINT_GRAPH_BINDINGS,
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
    "qwen-image-2512:modular-image-to-image:v1": {
        "modelType": "QwenImageModularPipeline",
        "mode": "image_to_image",
        "profile": _MODULAR_CONTROL_PROFILE,
        "roles": _MODULAR_SDXL_EDIT_GRAPH_ROLES,
        "edges": _MODULAR_QWEN_IMAGE_TO_IMAGE_GRAPH_EDGES,
        "bindings": _MODULAR_QWEN_IMAGE_TO_IMAGE_GRAPH_BINDINGS,
        "autoRequirementKey": "QwenImageModularPipeline:image_to_image",
        "autoRequirements": _MODULAR_QWEN_IMAGE_AUTO_REQUIREMENTS,
    },
    "qwen-image-2512:modular-text-to-image:v1": {
        "modelType": "QwenImageModularPipeline",
        "mode": "modular_text_to_image",
        "profile": _MODULAR_CONTROL_PROFILE,
        "roles": _MODULAR_TEXT_TO_IMAGE_GRAPH_ROLES,
        "edges": _MODULAR_QWEN_TEXT_TO_IMAGE_GRAPH_EDGES,
        "bindings": _MODULAR_QWEN_TEXT_TO_IMAGE_GRAPH_BINDINGS,
        "autoRequirements": _MODULAR_QWEN_IMAGE_AUTO_REQUIREMENTS,
    },
    "qwen-image-2512:modular-inpainting:v1": {
        "modelType": "QwenImageModularPipeline",
        "mode": "inpainting",
        "profile": _MODULAR_CONTROL_PROFILE,
        "roles": _MODULAR_SDXL_INPAINT_GRAPH_ROLES,
        "edges": _MODULAR_QWEN_INPAINT_GRAPH_EDGES,
        "bindings": _MODULAR_QWEN_INPAINT_GRAPH_BINDINGS,
        "autoRequirementKey": "QwenImageModularPipeline:inpainting",
        "autoRequirements": _MODULAR_QWEN_IMAGE_AUTO_REQUIREMENTS,
    },
    "qwen-image-2512:modular-control-image-to-image:v1": {
        "modelType": "QwenImageModularPipeline",
        "mode": "control_edit_image",
        "profile": _MODULAR_CONTROL_PROFILE,
        "roles": _MODULAR_SDXL_CONTROL_EDIT_GRAPH_ROLES,
        "edges": _MODULAR_QWEN_CONTROL_EDIT_GRAPH_EDGES,
        "bindings": _MODULAR_QWEN_CONTROL_EDIT_GRAPH_BINDINGS,
        "autoRequirementKey": "QwenImageModularPipeline:control_edit_image",
        "autoRequirements": _MODULAR_QWEN_IMAGE_AUTO_REQUIREMENTS,
    },
    "qwen-image-2512:modular-control-inpainting:v1": {
        "modelType": "QwenImageModularPipeline",
        "mode": "control_inpaint",
        "profile": _MODULAR_CONTROL_PROFILE,
        "roles": _MODULAR_SDXL_CONTROL_INPAINT_GRAPH_ROLES,
        "edges": _MODULAR_QWEN_CONTROL_INPAINT_GRAPH_EDGES,
        "bindings": _MODULAR_QWEN_CONTROL_INPAINT_GRAPH_BINDINGS,
        "autoRequirementKey": "QwenImageModularPipeline:control_inpaint",
        "autoRequirements": _MODULAR_QWEN_IMAGE_AUTO_REQUIREMENTS,
    },
    "qwen-image-controlnet-direct:control-image:v1": {
        "modelType": "QwenImageControlNetPipeline",
        "mode": "control_image",
        "profile": _QWEN_DIRECT_CONTROL_PROFILE,
        "roles": _CONTROL_GRAPH_ROLES,
        "edges": _CONTROL_GRAPH_EDGES,
        "bindings": _QWEN_DIRECT_CONTROL_GRAPH_BINDINGS,
        "capability": {
            "modelType": "QwenImageControlNetPipeline",
            "label": "Qwen Image ControlNet (Standard Diffusers)",
            "displayName": "Qwen-Image-2512 + ControlNet Union",
            "family": "Qwen Image",
            "supportTier": "supported",
            "qualificationStatus": "graph-qualified-execution-pending",
            "qualifiedModes": [],
            "defaultRepo": QWEN_IMAGE_2512_REPO,
            "downloadFiles": QWEN_IMAGE_2512_DIFFUSERS_FILES,
            "artifactLabel": "Reused bfloat16 safetensors base plus pinned ControlNet safetensors component",
            "defaultDtype": "bfloat16",
            "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
            "recommendedSteps": 50,
            "recommendedGuidance": 4.5,
            "guidanceLabel": "Guidance",
            "supportsNegativePrompt": True,
            "supportsImageInput": True,
            "supportsMask": False,
            "supportsMultiImage": False,
            "supportsControlImage": True,
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
                "steps": 28,
                "width": 768,
                "height": 768,
            },
            "modes": ["control_image"],
            "modeRequirements": {
                "control_image": {
                    "modelRequirements": studio_model_requirements_for_pair(
                        "QwenImageControlNetPipeline", "control_image"
                    ),
                    "requiredImages": ["controlImage"],
                    "note": "Requires the pinned Qwen ControlNet Union component and one prepared control image.",
                }
            },
            "additionalRequirements": studio_model_requirements_for_pair(
                "QwenImageControlNetPipeline", "control_image"
            ),
            "revisionCandidates": [require_catalog_revision(QWEN_IMAGE_2512_REPO)],
            "executionStatus": "expert_only",
            "autoEligible": False,
            "templateEligible": True,
            "galleryEligible": False,
            "liveProof": False,
            "notes": [
                "This separate standard-pipeline pair does not replace the approved Modular Qwen control workflow.",
                "Auto, Gallery, and live qualification remain disabled pending reviewed generated assets.",
            ],
        },
    },
    "qwen-image-layered-direct:layer-decomposition:v1": {
        "modelType": "QwenImageLayeredPipeline",
        "mode": "layer_decomposition",
        "profile": _QWEN_DIRECT_LAYERED_PROFILE,
        "roles": _LAYER_DECOMPOSITION_GRAPH_ROLES,
        "edges": _LAYER_DECOMPOSITION_GRAPH_EDGES,
        "bindings": _LAYER_DECOMPOSITION_GRAPH_BINDINGS,
        "capability": {
            "modelType": "QwenImageLayeredPipeline",
            "label": "Qwen Image Layered (Standard Diffusers)",
            "displayName": "Qwen-Image-Layered",
            "family": "Qwen Image",
            "supportTier": "supported",
            "qualificationStatus": "graph-qualified-execution-pending",
            "qualifiedModes": [],
            "defaultRepo": "Qwen/Qwen-Image-Layered",
            "downloadFiles": QWEN_IMAGE_LAYERED_DIFFUSERS_FILES,
            "artifactLabel": "Reused bfloat16 safetensors Diffusers repository",
            "defaultDtype": "bfloat16",
            "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "source"},
            "recommendedSteps": 50,
            "recommendedGuidance": 4.0,
            "guidanceLabel": "Guidance",
            "supportsNegativePrompt": True,
            "supportsImageInput": True,
            "supportsMask": False,
            "supportsMultiImage": False,
            "supportsControlImage": False,
            "supportsLayers": True,
            "supportsLora": True,
            "layerCount": {"default": 4, "min": 1, "max": 10},
            "layerResolutions": [640, 1024],
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
                "steps": 30,
                "resolution": 640,
            },
            "modes": ["layer_decomposition"],
            "modeRequirements": {
                "layer_decomposition": {
                    "requiredImages": ["referenceImages"],
                    "note": "Requires exactly one source image and emits one ordered layer stack.",
                }
            },
            "revisionCandidates": [require_catalog_revision("Qwen/Qwen-Image-Layered")],
            "executionStatus": "expert_only",
            "autoEligible": False,
            "templateEligible": True,
            "galleryEligible": False,
            "liveProof": False,
            "notes": [
                "This separate standard-pipeline pair does not replace the approved Modular Qwen layered workflow.",
                "Auto, Gallery, and live qualification remain disabled pending reviewed generated assets.",
            ],
        },
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
            "downloadFiles": STABLE_AUDIO_DIFFUSERS_FILES,
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
    **_HELIOS_STUDIO_EXECUTION_SPEC_DEFINITIONS,
    **_HUNYUAN_VIDEO_15_STUDIO_EXECUTION_SPEC_DEFINITIONS,
    **_MINIMAX_H3_STUDIO_EXECUTION_SPEC_DEFINITIONS,
    **_WAN_ANIMATE_2_STUDIO_EXECUTION_SPEC_DEFINITIONS,
    **_COSMOS3_NANO_STUDIO_EXECUTION_SPEC_DEFINITIONS,
    **_COSMOS3_DISTILLED_STUDIO_EXECUTION_SPEC_DEFINITIONS,
    "anima:modular-text-to-image:v1": {
        "modelType": "AnimaModularPipeline",
        "mode": "text_to_image",
        "profile": deepcopy(_ANIMA_MODULAR_PROFILE),
        "capability": deepcopy(_ANIMA_MODULAR_CAPABILITY),
        "roles": _MODULAR_WHOLE_IMAGE_TEXT_GRAPH_ROLES,
        "edges": _MODULAR_WHOLE_IMAGE_TEXT_GRAPH_EDGES,
        "bindings": _MODULAR_WHOLE_IMAGE_TEXT_GRAPH_BINDINGS,
    },
    "anima:modular-image-to-image:v1": {
        "modelType": "AnimaModularPipeline",
        "mode": "image_to_image",
        "profile": deepcopy(_ANIMA_MODULAR_PROFILE),
        "capability": deepcopy(_ANIMA_MODULAR_CAPABILITY),
        "roles": _MODULAR_WHOLE_IMAGE_EDIT_GRAPH_ROLES,
        "edges": _MODULAR_WHOLE_IMAGE_EDIT_GRAPH_EDGES,
        "bindings": _MODULAR_WHOLE_IMAGE_EDIT_GRAPH_BINDINGS,
    },
    "minimax-music3:modular-text-to-audio:v1": {
        "modelType": "MiniMaxMusic3ModularPipeline",
        "mode": "text_to_audio",
        "profile": {
            "id": "minimax-music3:official-modular-workflow",
            "model_type": "MiniMaxMusic3ModularPipeline",
            "modes": ("text_to_audio",),
            "loader_module": "modules.ModularDiffusers",
            "loader_action": "ModelsLoader",
            "execution_path": "modular-diffusers",
            "pipeline_class": "MiniMaxMusic3ModularPipeline",
            "default_repo": MINIMAX_MUSIC3_REPO,
            "fallback_repo": None,
            "quantizable_components": (),
            "default_quantized_components": (),
            "supported_offload_modes": (
                OFFLOAD_MODE_NONE,
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ),
            "retry_offload_modes": (OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
            "max_low_memory_side": None,
            "max_low_memory_steps": 30,
            "live_proof": False,
            "compatible_repos": (),
        },
        "capability": {
            "modelType": "MiniMaxMusic3ModularPipeline",
            "label": "MiniMax Music 3 (Modular Diffusers)",
            "displayName": "MiniMax-Music3",
            "family": "MiniMax Music 3",
            "supportTier": "supported",
            "qualificationStatus": "graph-qualified-execution-pending",
            "qualifiedModes": [],
            "defaultRepo": MINIMAX_MUSIC3_REPO,
            "downloadFiles": MINIMAX_MUSIC3_DIFFUSERS_FILES,
            "artifactLabel": "Exact reviewed Modular Diffusers component snapshot",
            "license": "MiniMax-Music3 Community License",
            "defaultDtype": "bfloat16",
            "defaultSize": {"width": 0, "height": 0, "aspectRatio": "custom"},
            "recommendedSteps": 30,
            "recommendedGuidance": 1.0,
            "guidanceLabel": "Official workflow guidance",
            "supportsImageInput": False,
            "supportsMask": False,
            "supportsMultiImage": False,
            "supportsControlImage": False,
            "supportsLayers": False,
            "supportsLora": False,
            "supportsAudioInput": False,
            "outputKind": "audio",
            "recommendedSampleRate": 44100,
            "recommendedDuration": 60,
            "offloadSupport": {
                "default": OFFLOAD_MODE_GROUP_CPU,
                "lowVram": OFFLOAD_MODE_GROUP_CPU,
                "emergency": OFFLOAD_MODE_GROUP_DISK,
                "modes": [
                    OFFLOAD_MODE_NONE,
                    OFFLOAD_MODE_MODEL_CPU,
                    OFFLOAD_MODE_GROUP_CPU,
                    OFFLOAD_MODE_GROUP_DISK,
                ],
            },
            "lowVram": {
                "dtype": "bfloat16",
                "autoOffload": True,
                "offloadMode": OFFLOAD_MODE_GROUP_CPU,
                "steps": 30,
            },
            "modes": ["text_to_audio"],
            "executionStatus": "expert_only",
            "revisionCandidates": [
                require_catalog_revision(MINIMAX_MUSIC3_REPO, model_type="MiniMaxMusic3ModularPipeline")
            ],
            "autoEligible": False,
            "templateEligible": False,
            "galleryEligible": False,
            "liveProof": False,
            "notes": [
                "Runs the official semantic_generator, denoise, and decode Modular blocks without reimplementing upstream loops.",
                "The MiniMax-Music3 Community License requires revision-bound acknowledgement before install and Run.",
                "Auto and Gallery remain disabled until real CUDA execution and manual output review are complete.",
            ],
        },
        "roles": _MODULAR_WHOLE_AUDIO_GRAPH_ROLES,
        "edges": _MODULAR_WHOLE_AUDIO_GRAPH_EDGES,
        "bindings": _MODULAR_WHOLE_AUDIO_GRAPH_BINDINGS,
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
            "downloadFiles": LONGCAT_AUDIO_DIT_DIFFUSERS_FILES,
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
            "downloadFiles": AUDIO_LDM2_DIFFUSERS_FILES,
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
            "downloadFiles": SHAP_E_SAFE_COMPONENT_FILES,
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
    "shap-e:image-to-3d:v1": {
        "modelType": "ShapEImg2ImgPipeline",
        "mode": "image_to_3d",
        "profile": {
            "id": "shap-e-img2img:direct",
            "model_type": "ShapEImg2ImgPipeline",
            "modes": ("image_to_3d",),
            "loader_module": "modules.DiffusersThreeD",
            "loader_action": "LoadPipeline",
            "execution_path": "direct-diffusers-three-d",
            "pipeline_class": "ShapEImg2ImgPipeline",
            "default_repo": SHAP_E_IMG2IMG_REPO,
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
            "modelType": "ShapEImg2ImgPipeline",
            "label": "Shap-E Image to Rendered 3D",
            "displayName": "shap-e-img2img",
            "family": "Shap-E",
            "supportTier": "supported",
            "qualificationStatus": "graph-qualified-execution-pending",
            "qualifiedModes": [],
            "defaultRepo": SHAP_E_IMG2IMG_REPO,
            "artifactLabel": "Explicit safe-component Diffusers assembly",
            "downloadFiles": SHAP_E_IMG2IMG_SAFE_COMPONENT_FILES,
            "defaultDtype": "float16",
            "defaultSize": {"width": 256, "height": 256, "aspectRatio": "1:1"},
            "recommendedSteps": 64,
            "recommendedGuidance": 3.0,
            "guidanceLabel": "Guidance",
            "supportsImageInput": True,
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
            "modes": ["image_to_3d"],
            "modeRequirements": {"image_to_3d": {"requiredImages": ["referenceImages"]}},
            "executionStatus": "expert_only",
            "revisionCandidates": [require_catalog_revision(SHAP_E_IMG2IMG_REPO, model_type="ShapEImg2ImgPipeline")],
            "autoEligible": False,
            "templateEligible": True,
            "galleryEligible": False,
            "notes": [
                "The generic contract accepts exactly one bounded source image and returns a rendered orbit; it does not expose latent or mesh serialization.",
                "The loader explicitly assembles only the official fp16 safetensors prior, CLIP vision encoder, processor, and pre-rename renderer component.",
                "MIT applies; Auto and Gallery remain disabled pending app-only installation, remote runtime, output-quality, and export review.",
            ],
        },
        "roles": _THREE_D_IMAGE_GRAPH_ROLES,
        "edges": _THREE_D_IMAGE_GRAPH_EDGES,
        "bindings": _THREE_D_IMAGE_GRAPH_BINDINGS,
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
        "bindings": _EDIT_GRAPH_BINDINGS + (("diffusersImagePipeline", "revision", "defaultRevision"),),
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
        "bindings": _INPAINT_GRAPH_BINDINGS + (("diffusersImagePipeline", "revision", "defaultRevision"),),
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
            "downloadFiles": SDXL_BASE_FP16_DIFFUSERS_FILES,
            "artifactLabel": "Diffusers fp16 safetensors repo",
            "defaultDtype": "float16",
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
                "dtype": "float16",
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

_SDXL_MODULAR_CAPABILITY = STUDIO_EXECUTION_SPEC_DEFINITIONS["sdxl-base:modular-text-to-image:v1"]["capability"]
for _mode in _SDXL_UNION_MODES:
    _route = _mode.removeprefix("control_union_")
    _required = ["controlImage"]
    if _route == "edit_image":
        _required.insert(0, "referenceImages")
    elif _route == "inpaint":
        _required[0:0] = ["referenceImages", "maskImage"]
    _SDXL_MODULAR_CAPABILITY["modeRequirements"][_mode] = {
        "requiredImages": _required,
        "modelRequirements": studio_model_requirements_for_pair("StableDiffusionXLModularPipeline", _mode),
        "note": "Requires a prepared control image and an explicit ControlNet Union control mode.",
    }
for _mode in (*_SDXL_IP_ADAPTER_MODES, *_SDXL_IP_ADAPTER_CONTROL_MODES, *_SDXL_IP_ADAPTER_UNION_MODES):
    _required = ["ipAdapterImage"]
    if "edit_image" in _mode:
        _required.insert(0, "referenceImages")
    elif _mode.endswith("inpaint"):
        _required[0:0] = ["referenceImages", "maskImage"]
    if "control" in _mode:
        _required.append("controlImage")
    _SDXL_MODULAR_CAPABILITY["modeRequirements"][_mode] = {
        "requiredImages": _required,
        "modelRequirements": studio_model_requirements_for_pair("StableDiffusionXLModularPipeline", _mode),
        "note": "Requires a separate IP-Adapter reference image; expanded adapter blocks remain independently editable.",
    }

STUDIO_EXECUTION_SPEC_DEFINITIONS["sdxl-base:modular-text-to-image:v1"] = {
    **STUDIO_EXECUTION_SPEC_DEFINITIONS["sdxl-base:modular-text-to-image:v1"],
    "capability": _SDXL_MODULAR_CAPABILITY,
}

for _spec_name, _mode, _route, _control, _ip_adapter in (
    ("controlnet-union-text-to-image", "control_union_image", "text2image", "union", False),
    ("controlnet-union-image-to-image", "control_union_edit_image", "image2image", "union", False),
    ("controlnet-union-inpainting", "control_union_inpaint", "inpainting", "union", False),
    ("ip-adapter-text-to-image", "ip_adapter_image", "text2image", None, True),
    ("ip-adapter-image-to-image", "ip_adapter_edit_image", "image2image", None, True),
    ("ip-adapter-inpainting", "ip_adapter_inpaint", "inpainting", None, True),
    ("ip-adapter-controlnet-text-to-image", "ip_adapter_control_image", "text2image", "ordinary", True),
    ("ip-adapter-controlnet-image-to-image", "ip_adapter_control_edit_image", "image2image", "ordinary", True),
    ("ip-adapter-controlnet-inpainting", "ip_adapter_control_inpaint", "inpainting", "ordinary", True),
    (
        "ip-adapter-controlnet-union-text-to-image",
        "ip_adapter_control_union_image",
        "text2image",
        "union",
        True,
    ),
    (
        "ip-adapter-controlnet-union-image-to-image",
        "ip_adapter_control_union_edit_image",
        "image2image",
        "union",
        True,
    ),
    (
        "ip-adapter-controlnet-union-inpainting",
        "ip_adapter_control_union_inpaint",
        "inpainting",
        "union",
        True,
    ),
):
    _roles, _edges, _bindings = _MODULAR_SDXL_REMAINING_GRAPHS[(_route, _control, _ip_adapter)]
    STUDIO_EXECUTION_SPEC_DEFINITIONS[f"sdxl-base:modular-{_spec_name}:v1"] = {
        "modelType": "StableDiffusionXLModularPipeline",
        "mode": _mode,
        "profile": _MODULAR_SDXL_PROFILE,
        "roles": _roles,
        "edges": _edges,
        "bindings": _bindings,
    }

_FLUX_COMBINED_CONTROL_DEFINITIONS = {
    _spec_id: STUDIO_EXECUTION_SPEC_DEFINITIONS.pop(_spec_id)
    for _spec_id in (
        "flux-depth:control-edit-image:v1",
        "flux-depth:control-inpaint:v1",
        "flux-canny:control-edit-image:v1",
        "flux-canny:control-inpaint:v1",
    )
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
    *,
    download_files: list[str] | None = None,
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
        **({"downloadFiles": download_files} if download_files else {}),
        "artifactLabel": "Diffusers video repo",
        "defaultDtype": "bfloat16",
        "defaultSize": {"width": 768, "height": 512, "aspectRatio": "custom"},
        "recommendedSteps": 30,
        "recommendedGuidance": 3.0,
        "guidanceLabel": "Guidance",
        "supportsImageInput": any(contract.get("requiredImages") for contract in (input_contracts or {}).values()),
        "supportsMask": False,
        "supportsMultiImage": False,
        "supportsControlImage": False,
        "supportsLayers": False,
        "supportsLora": False,
        "supportsVideoInput": any(contract.get("requiredVideos") for contract in (input_contracts or {}).values()),
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
        "downloadFiles": SD15_SHARED_DIFFUSERS_FILES,
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


def _animatediff_extended_capability(
    model_type: str,
    mode: str,
    *,
    required_videos: tuple[str, ...] = (),
    controlnet: bool = False,
    pag: bool = False,
) -> dict[str, Any]:
    capability = deepcopy(_animatediff_capability("AnimateDiffPipeline", lcm=False))
    suffix = {
        "AnimateDiffPAGPipeline": " + PAG",
        "AnimateDiffVideoToVideoPipeline": " video-to-video",
        "AnimateDiffControlNetPipeline": " + ControlNet",
        "AnimateDiffVideoToVideoControlNetPipeline": " video-to-video + ControlNet",
    }[model_type]
    requirements = studio_model_requirements_for_pair(model_type, mode)
    mode_requirements: dict[str, Any] = {
        "modelRequirements": requirements,
        "note": (
            "Uses the exact AnimateDiff SD1.5 v2 motion module"
            + (" and pinned SD1.5 Canny ControlNet" if controlnet else "")
            + "."
        ),
    }
    if required_videos:
        mode_requirements["requiredVideos"] = list(required_videos)
    capability.update(
        {
            "modelType": model_type,
            "label": f"AnimateDiff SD1.5 v2{suffix}",
            "displayName": f"AnimateDiff motion adapter v1.5.2{suffix}",
            "artifactLabel": (
                "SD1.5 safetensors base plus pinned fp16 safetensors motion adapter"
                + (" and pinned safetensors Canny ControlNet" if controlnet else "")
            ),
            "supportsVideoInput": bool(required_videos),
            "modes": [mode],
            "modeRequirements": {mode: mode_requirements},
            "additionalRequirements": requirements,
            "liveProof": False,
        }
    )
    if mode in {"video_to_video", "control_video_to_video"}:
        capability["recommendedStrength"] = 0.8
    if controlnet:
        capability["conditioningScale"] = 1.0
    if pag:
        capability["recommendedPagScale"] = 3.0
        capability["recommendedPagAdaptiveScale"] = 0.0
    capability["notes"] = [
        "The SD1.5 base, motion adapter, and any ControlNet component are pinned independently and load only safetensors weights.",
        "The motion repository does not declare a weight license; users must establish authorization before use.",
        "Auto and Gallery publication remain disabled until exact remote runtime, quality, and rights proof is reviewed.",
    ]
    return capability


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
        "downloadFiles": COGVIDEOX_2B_DIFFUSERS_FILES,
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


def _cogvideox_v2v_capability() -> dict[str, Any]:
    capability = deepcopy(_cogvideox_capability())
    capability.update(
        {
            "modelType": "CogVideoXVideoToVideoPipeline",
            "label": "CogVideoX-2B Video-to-Video",
            "displayName": "CogVideoX-2B Video-to-Video",
            "supportsVideoInput": True,
            "modes": ["video_to_video"],
            "modeRequirements": {
                "video_to_video": {
                    "requiredVideos": ["sourceVideo"],
                    "note": "Uses one normalized source video with the exact CogVideoX-2B safetensors snapshot.",
                }
            },
            "recommendedStrength": 0.8,
            "liveProof": False,
        }
    )
    capability["notes"] = [
        "The admitted source graph is bounded to 9-25 frames in 4k+1 form at the native 720x480 size.",
        "The source video is normalized before the standard Diffusers video-to-video route.",
        "Auto and Gallery publication remain disabled until exact remote runtime and quality proof is reviewed.",
    ]
    return capability


def _allegro_capability() -> dict[str, Any]:
    return {
        "modelType": "AllegroPipeline",
        "label": "Allegro",
        "displayName": "Allegro",
        "family": "Allegro",
        "supportTier": "supported",
        "qualificationStatus": "graph-qualified-execution-pending",
        "qualifiedModes": [],
        "defaultRepo": ALLEGRO_REPO,
        "artifactLabel": "Official Apache-2.0 safetensors Diffusers repo",
        "downloadFiles": ALLEGRO_DIFFUSERS_FILES,
        "defaultDtype": "bfloat16",
        "defaultSize": {"width": 1280, "height": 720, "aspectRatio": "16:9"},
        "recommendedSteps": 100,
        "recommendedGuidance": 7.5,
        "recommendedMaxSequenceLength": 512,
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
        "recommendedFrames": 88,
        "recommendedFps": 15,
        "offloadSupport": {
            "default": OFFLOAD_MODE_SEQUENTIAL_CPU,
            "lowVram": OFFLOAD_MODE_SEQUENTIAL_CPU,
            "emergency": OFFLOAD_MODE_GROUP_DISK,
            "modes": list(_DIRECT_OFFLOAD_MODES),
        },
        "lowVram": {
            "dtype": "bfloat16",
            "autoOffload": True,
            "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
            "steps": 100,
            "width": 1280,
            "height": 720,
            "numFrames": 88,
        },
        "modes": ["text_to_video"],
        "modeRequirements": {
            "text_to_video": {
                "note": "Uses the exact Allegro safetensors snapshot with an FP32 tiled VAE and sequential CPU offload."
            }
        },
        "executionStatus": "expert_only",
        "revisionCandidates": [require_catalog_revision(ALLEGRO_REPO)],
        "autoEligible": False,
        "templateEligible": True,
        "galleryEligible": False,
        "notes": [
            "The admitted source graph preserves the native 88-frame 1280x720 recipe at 15 FPS.",
            "The duplicate PyTorch .bin text-encoder shards are excluded in favor of the complete safetensors index.",
            "Auto and Gallery publication remain disabled until exact remote runtime, safety, and quality proof is reviewed.",
        ],
    }


def _latte_capability() -> dict[str, Any]:
    return {
        "modelType": "LattePipeline",
        "label": "Latte",
        "displayName": "Latte",
        "family": "Latte",
        "supportTier": "supported",
        "qualificationStatus": "graph-qualified-execution-pending",
        "qualifiedModes": [],
        "defaultRepo": LATTE_REPO,
        "artifactLabel": "Official Apache-2.0 safetensors Diffusers repo",
        "downloadFiles": LATTE_DIFFUSERS_FILES,
        "defaultDtype": "float16",
        "defaultSize": {"width": 512, "height": 512, "aspectRatio": "1:1"},
        "recommendedSteps": 50,
        "recommendedGuidance": 7.5,
        "recommendedMaxSequenceLength": 120,
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
            "default": OFFLOAD_MODE_SEQUENTIAL_CPU,
            "lowVram": OFFLOAD_MODE_SEQUENTIAL_CPU,
            "emergency": OFFLOAD_MODE_GROUP_DISK,
            "modes": list(_DIRECT_OFFLOAD_MODES),
        },
        "lowVram": {
            "dtype": "float16",
            "autoOffload": True,
            "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
            "steps": 50,
            "width": 512,
            "height": 512,
            "numFrames": 16,
        },
        "modes": ["text_to_video"],
        "modeRequirements": {
            "text_to_video": {
                "note": "Uses the exact Latte safetensors snapshot and its native 16-frame recipe with sequential CPU offload."
            }
        },
        "executionStatus": "expert_only",
        "revisionCandidates": [require_catalog_revision(LATTE_REPO)],
        "autoEligible": False,
        "templateEligible": True,
        "galleryEligible": False,
        "notes": [
            "The admitted source graph is fixed to the native 16-frame 512x512 recipe at 8 FPS.",
            "The unsafe legacy .pt checkpoint and unreferenced optional temporal VAE are excluded from execution.",
            "Auto and Gallery publication remain disabled until exact remote runtime, safety, and quality proof is reviewed.",
        ],
    }


def _mochi_capability() -> dict[str, Any]:
    return {
        "modelType": "MochiPipeline",
        "label": "Mochi 1 Preview",
        "displayName": "Mochi 1 Preview",
        "family": "Mochi",
        "supportTier": "supported",
        "qualificationStatus": "graph-qualified-execution-pending",
        "qualifiedModes": [],
        "defaultRepo": MOCHI_REPO,
        "artifactLabel": "Official Apache-2.0 BF16 safetensors Diffusers repo",
        "downloadFiles": MOCHI_DIFFUSERS_FILES,
        "defaultDtype": "bfloat16",
        "defaultSize": {"width": 848, "height": 480, "aspectRatio": "16:9"},
        "recommendedSteps": 64,
        "recommendedGuidance": 4.5,
        "recommendedMaxSequenceLength": 256,
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
        "recommendedFrames": 31,
        "recommendedFps": 30,
        "offloadSupport": {
            "default": OFFLOAD_MODE_SEQUENTIAL_CPU,
            "lowVram": OFFLOAD_MODE_SEQUENTIAL_CPU,
            "emergency": OFFLOAD_MODE_GROUP_DISK,
            "modes": list(_DIRECT_OFFLOAD_MODES),
        },
        "lowVram": {
            "dtype": "bfloat16",
            "autoOffload": True,
            "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
            "steps": 64,
            "width": 848,
            "height": 480,
            "numFrames": 31,
        },
        "modes": ["text_to_video"],
        "modeRequirements": {
            "text_to_video": {
                "note": "Uses the exact Mochi BF16 variant, indexed T5 encoder, native 31-frame recipe, and mandatory VAE tiling."
            }
        },
        "executionStatus": "expert_only",
        "revisionCandidates": [require_catalog_revision(MOCHI_REPO)],
        "autoEligible": False,
        "templateEligible": True,
        "galleryEligible": False,
        "notes": [
            "The admitted source graph is fixed to the publisher's 31-frame 848x480 recipe at 30 FPS.",
            "Duplicate flat-format, FP32, and unindexed text-encoder artifacts are excluded from execution.",
            "Auto and Gallery publication remain disabled until exact remote runtime, safety, and quality proof is reviewed.",
        ],
    }


def _sana_video_capability(*, image_conditioned: bool) -> dict[str, Any]:
    model_type = "SanaImageToVideoPipeline" if image_conditioned else "SanaVideoPipeline"
    mode = "image_to_video" if image_conditioned else "text_to_video"
    label = "SANA-Video 2B 480p I2V" if image_conditioned else "SANA-Video 2B 480p"
    requirements = (
        {
            "requiredImages": ["referenceImages"],
            "note": "Requires exactly one opening image and preserves it as the first latent frame.",
        }
        if image_conditioned
        else {"note": "Uses the native text-only recipe with a backend-owned motion-score suffix of 30."}
    )
    return {
        "modelType": model_type,
        "label": label,
        "displayName": label,
        "family": "SANA Video",
        "supportTier": "supported",
        "qualificationStatus": "graph-qualified-execution-pending",
        "qualifiedModes": [],
        "defaultRepo": SANA_VIDEO_REPO,
        "downloadFiles": SANA_VIDEO_DIFFUSERS_FILES,
        "artifactLabel": "Official Apache-2.0 mixed-precision safetensors Diffusers repo",
        "defaultDtype": "bfloat16",
        "defaultSize": {"width": 832, "height": 480, "aspectRatio": "16:9"},
        "recommendedSteps": 50,
        "recommendedGuidance": 6.0,
        "recommendedMaxSequenceLength": 300,
        "guidanceLabel": "Guidance",
        "supportsImageInput": image_conditioned,
        "supportsMask": False,
        "supportsMultiImage": False,
        "supportsControlImage": False,
        "supportsLayers": False,
        "supportsLora": False,
        "supportsVideoInput": False,
        "supportsVideoMask": False,
        "outputKind": "video",
        "recommendedFrames": 81,
        "recommendedFps": 16,
        "offloadSupport": {
            "default": OFFLOAD_MODE_SEQUENTIAL_CPU,
            "lowVram": OFFLOAD_MODE_SEQUENTIAL_CPU,
            "emergency": OFFLOAD_MODE_GROUP_DISK,
            "modes": list(_DIRECT_OFFLOAD_MODES),
        },
        "lowVram": {
            "dtype": "bfloat16",
            "autoOffload": True,
            "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
            "steps": 50,
            "width": 832,
            "height": 480,
            "numFrames": 81,
        },
        "modes": [mode],
        "modeRequirements": {mode: requirements},
        "executionStatus": "expert_only",
        "revisionCandidates": [require_catalog_revision(SANA_VIDEO_REPO)],
        "autoEligible": False,
        "templateEligible": True,
        "galleryEligible": False,
        "notes": [
            "The admitted source graph is fixed to the publisher's 832x480, 81-frame, 50-step recipe at 16 FPS.",
            "The transformer and text encoder run in BF16 while the Wan VAE remains FP32 with mandatory tiling.",
            "Auto and Gallery remain disabled until exact remote runtime, safety, and quality proof is reviewed.",
        ],
    }


_WAN_ANIMATE_MODES = ("character_animate", "character_replace")
_LTX2_MODES = ("text_to_video", "image_to_video", "video_to_video", "reference_to_video")
_P2_VIDEO_PROFILES = {
    "wan-1.3b-modular": _planning_video_profile(
        "wan-1.3b:modular-text-to-video",
        "WanModularPipeline",
        ("text_to_video",),
        "WanModularPipeline",
        WAN_T2V_1_3B_REPO,
        loader_module="modules.ModularDiffusers",
        loader_action="ModelsLoader",
        execution_path="modular-diffusers",
    ),
    "wan-i2v-480p-modular": _planning_video_profile(
        "wan-i2v-480p:modular",
        "WanImage2VideoModularPipeline",
        ("single_image_to_video",),
        "WanImage2VideoModularPipeline",
        WAN_I2V_14B_480P_REPO,
        loader_module="modules.ModularDiffusers",
        loader_action="ModelsLoader",
        execution_path="modular-diffusers",
    ),
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
    "ltx2-in-context": _planning_video_profile(
        "ltx2-in-context:direct",
        "LTX2InContextPipeline",
        ("in_context_to_video",),
        "LTX2InContextPipeline",
        LTX2_REPO,
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
    "animatediff-pag": _planning_video_profile(
        "animatediff-sd15-v2-pag:direct",
        "AnimateDiffPAGPipeline",
        ("text_to_video",),
        "AnimateDiffPAGPipeline",
        SD15_BASE_REPO,
    ),
    "animatediff-video-to-video": _planning_video_profile(
        "animatediff-sd15-v2-video-to-video:direct",
        "AnimateDiffVideoToVideoPipeline",
        ("video_to_video",),
        "AnimateDiffVideoToVideoPipeline",
        SD15_BASE_REPO,
    ),
    "animatediff-controlnet": _planning_video_profile(
        "animatediff-sd15-v2-controlnet:direct",
        "AnimateDiffControlNetPipeline",
        ("control_to_video",),
        "AnimateDiffControlNetPipeline",
        SD15_BASE_REPO,
    ),
    "animatediff-controlnet-video-to-video": _planning_video_profile(
        "animatediff-sd15-v2-controlnet-video-to-video:direct",
        "AnimateDiffVideoToVideoControlNetPipeline",
        ("control_video_to_video",),
        "AnimateDiffVideoToVideoControlNetPipeline",
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
    "cogvideox-2b-video-to-video": _planning_video_profile(
        "cogvideox-2b-video-to-video:direct",
        "CogVideoXVideoToVideoPipeline",
        ("video_to_video",),
        "CogVideoXVideoToVideoPipeline",
        COGVIDEOX_2B_REPO,
    ),
    "allegro": _planning_video_profile(
        "allegro:direct",
        "AllegroPipeline",
        ("text_to_video",),
        "AllegroPipeline",
        ALLEGRO_REPO,
    ),
    "latte": _planning_video_profile(
        "latte:direct",
        "LattePipeline",
        ("text_to_video",),
        "LattePipeline",
        LATTE_REPO,
    ),
    "mochi": _planning_video_profile(
        "mochi:direct",
        "MochiPipeline",
        ("text_to_video",),
        "MochiPipeline",
        MOCHI_REPO,
    ),
    "sana-video-480p": _planning_video_profile(
        "sana-video-480p:direct",
        "SanaVideoPipeline",
        ("text_to_video",),
        "SanaVideoPipeline",
        SANA_VIDEO_REPO,
    ),
    "sana-video-480p-i2v": _planning_video_profile(
        "sana-video-480p-i2v:direct",
        "SanaImageToVideoPipeline",
        ("image_to_video",),
        "SanaImageToVideoPipeline",
        SANA_VIDEO_REPO,
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
_WAN_MODULAR_TEXT_CAPABILITY = _planning_video_capability(
    "WanModularPipeline",
    "Wan 2.1 1.3B Text to Video (Modular Diffusers)",
    "Wan Video",
    WAN_T2V_1_3B_REPO,
    ("text_to_video",),
    download_files=WAN_T2V_1_3B_DIFFUSERS_FILES,
)
_WAN_MODULAR_TEXT_CAPABILITY.update(
    {
        "autoEligible": False,
        "templateEligible": True,
        "galleryEligible": False,
        "liveProof": False,
    }
)
_WAN_MODULAR_I2V_CAPABILITY = _planning_video_capability(
    "WanImage2VideoModularPipeline",
    "Wan 2.1 Image to Video (Modular Diffusers)",
    "Wan Video",
    WAN_I2V_14B_480P_REPO,
    ("single_image_to_video", "image_to_video"),
    {
        "single_image_to_video": {"requiredImages": ["referenceImages"]},
        "image_to_video": {"requiredImages": ["referenceImages", "lastImage"]},
    },
    download_files=WAN_I2V_14B_480P_DIFFUSERS_FILES,
)
_WAN_MODULAR_I2V_CAPABILITY.update(
    {
        "revisionCandidates": [
            require_catalog_revision(WAN_I2V_14B_480P_REPO),
            require_catalog_revision(WAN_FLF_REPO),
        ],
        "artifactSelections": [
            {
                "modes": ["single_image_to_video"],
                "repo": WAN_I2V_14B_480P_REPO,
                "revision": require_catalog_revision(WAN_I2V_14B_480P_REPO),
                "downloadFiles": WAN_I2V_14B_480P_DIFFUSERS_FILES,
                "label": "Wan 2.1 I2V 14B 480P Diffusers repo",
            },
            {
                "modes": ["image_to_video"],
                "repo": WAN_FLF_REPO,
                "revision": require_catalog_revision(WAN_FLF_REPO),
                "downloadFiles": WAN_FLF_14B_DIFFUSERS_FILES,
                "label": "Wan 2.1 FLF2V 14B 720P Diffusers repo",
            },
        ],
        "autoEligible": False,
        "templateEligible": True,
        "galleryEligible": False,
        "liveProof": False,
        "notes": [
            "The single-image and first/last-frame Cluster workflows share the exact Wan prompt, conditioning, denoise, and decode actions while retaining separate pinned repositories.",
            "Public execution and Auto selection remain disabled pending collapsed/expanded visible-frontend qualification and manual review.",
        ],
    }
)
_WAN_MODULAR_TEXT_AUTO_REQUIREMENTS = {
    "supportedTasks": ["text_to_video"],
    "defaultRepo": WAN_T2V_1_3B_REPO,
    "qualityDefaults": {
        "width": 832,
        "height": 480,
        "steps": 30,
        "guidanceScale": 5,
        "numFrames": 81,
    },
    "minimum": {
        "accelerator": "cuda",
        "vramBytes": 10 * _GIB,
        "systemRamBytes": 24 * _GIB,
        "diskFreeBytes": 20 * _GIB,
    },
    "recommended": {
        "accelerator": "cuda",
        "vramBytes": 13 * _GIB,
        "systemRamBytes": 32 * _GIB,
        "diskFreeBytes": 30 * _GIB,
    },
    "supportedOffloadModes": list(_DIRECT_OFFLOAD_MODES),
    "requiredPackages": ["diffusers", "transformers", "accelerate", "torch"],
    "guardedReason": (
        "Wan Modular text-to-video is planned per Cluster instance; changing its artifact or workflow invalidates "
        "only that Cluster's component and resource receipts."
    ),
}
_WAN_MODULAR_I2V_AUTO_REQUIREMENTS = {
    "supportedTasks": ["single_image_to_video"],
    "defaultRepo": WAN_I2V_14B_480P_REPO,
    "qualityDefaults": {
        "width": 832,
        "height": 480,
        "steps": 40,
        "guidanceScale": 5,
        "numFrames": 81,
    },
    "minimum": {
        "accelerator": "cuda",
        "vramBytes": 24 * _GIB,
        "systemRamBytes": 64 * _GIB,
        "diskFreeBytes": 70 * _GIB,
    },
    "recommended": {
        "accelerator": "cuda",
        "vramBytes": 48 * _GIB,
        "systemRamBytes": 96 * _GIB,
        "diskFreeBytes": 90 * _GIB,
    },
    "fullResidency": {
        "accelerator": "cuda",
        "vramBytes": 80 * _GIB,
        "systemRamBytes": 96 * _GIB,
    },
    "supportedOffloadModes": list(_DIRECT_OFFLOAD_MODES),
    "requiredPackages": ["diffusers", "transformers", "accelerate", "torch"],
    "guardedReason": (
        "Wan Modular image-to-video is a 14B workflow and is admitted conservatively until an exact live memory "
        "receipt is reviewed for this pinned artifact."
    ),
}

STUDIO_EXECUTION_SPEC_DEFINITIONS.update(
    {
        "wan-22-a14b:text-to-video:v1": {
            "modelType": "Wan22Pipeline",
            "mode": "text_to_video",
            "profile": _P2_VIDEO_PROFILES["wan22"],
            "capability": _planning_video_capability(
                "Wan22Pipeline",
                "Wan 2.2 T2V A14B",
                "Wan Video",
                WAN_22_T2V_A14B_REPO,
                ("text_to_video",),
                download_files=WAN_22_T2V_A14B_DIFFUSERS_FILES,
            ),
            "roles": _VIDEO_GRAPH_ROLES,
            "edges": _VIDEO_GRAPH_EDGES,
            "bindings": _VIDEO_REVISION_GRAPH_BINDINGS,
            "expertResourceRequirements": _WAN_22_A14B_EXPERT_RESOURCE_REQUIREMENTS,
        },
        "wan-animate:character-animate:v1": {
            "modelType": "WanAnimatePipeline",
            "mode": "character_animate",
            "profile": _P2_VIDEO_PROFILES["animate"],
            "capability": _planning_video_capability(
                "WanAnimatePipeline",
                "Wan 2.2 Animate",
                "Wan Video",
                WAN_ANIMATE_REPO,
                _WAN_ANIMATE_MODES,
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
                "LTXI2VLongMultiPromptPipeline",
                "LTX long-prompt I2V",
                "LTX Video",
                LTX_VIDEO_REPO,
                ("image_to_video",),
                {"image_to_video": {"requiredImages": ["referenceImages"]}},
                download_files=LTX_VIDEO_DIFFUSERS_FILES,
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
                "LTX2ConditionPipeline",
                "LTX-2 video and audio",
                "LTX Video",
                LTX2_REPO,
                _LTX2_MODES,
                _LTX2_INPUTS,
                download_files=LTX2_DIFFUSERS_FILES,
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
        "ltx2:in-context-reference-to-video:v1": {
            "modelType": "LTX2InContextPipeline",
            "mode": "in_context_to_video",
            "profile": _P2_VIDEO_PROFILES["ltx2-in-context"],
            "capability": _planning_video_capability(
                "LTX2InContextPipeline",
                "LTX-2 in-context Canny video and audio",
                "LTX Video",
                LTX2_REPO,
                ("in_context_to_video",),
                {
                    "in_context_to_video": {
                        "requiredVideos": ["referenceVideos"],
                        "note": "Requires one reference video preprocessed by the reviewed Canny IC-LoRA graph.",
                    }
                },
                download_files=LTX2_DIFFUSERS_FILES,
            ),
            "roles": _LTX2_IN_CONTEXT_GRAPH_ROLES,
            "edges": _LTX2_IN_CONTEXT_GRAPH_EDGES,
            "bindings": _LTX2_IN_CONTEXT_GRAPH_BINDINGS,
        },
        "framepack:image-to-video:v1": {
            "modelType": "HunyuanVideoFramepackPipeline",
            "mode": "image_to_video",
            "profile": _P2_VIDEO_PROFILES["framepack"],
            "capability": {
                **_planning_video_capability(
                    "HunyuanVideoFramepackPipeline",
                    "Hunyuan FramePack I2V",
                    "Wan Video",
                    FRAMEPACK_REPO,
                    ("image_to_video",),
                    {"image_to_video": {"requiredImages": ["referenceImages"]}},
                    download_files=FRAMEPACK_TRANSFORMER_DIFFUSERS_FILES,
                ),
                "additionalRequirements": studio_model_requirements_for_pair(
                    "HunyuanVideoFramepackPipeline", "image_to_video"
                ),
                "modeRequirements": {
                    "image_to_video": {
                        "modelRequirements": studio_model_requirements_for_pair(
                            "HunyuanVideoFramepackPipeline", "image_to_video"
                        ),
                        "requiredImages": ["referenceImages"],
                        "note": (
                            "Requires a source image plus the exact HunyuanVideo base and SigLIP "
                            "vision component selections."
                        ),
                    }
                },
            },
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
                "downloadFiles": STABLE_VIDEO_DIFFUSION_FP16_FILES,
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
        "animatediff-pag:text-to-video:v1": {
            "modelType": "AnimateDiffPAGPipeline",
            "mode": "text_to_video",
            "profile": _P2_VIDEO_PROFILES["animatediff-pag"],
            "capability": _animatediff_extended_capability(
                "AnimateDiffPAGPipeline",
                "text_to_video",
                pag=True,
            ),
            "roles": _VIDEO_GRAPH_ROLES,
            "edges": _VIDEO_GRAPH_EDGES,
            "bindings": _ANIMATEDIFF_PAG_GRAPH_BINDINGS,
        },
        "animatediff-video-to-video:video-to-video:v1": {
            "modelType": "AnimateDiffVideoToVideoPipeline",
            "mode": "video_to_video",
            "profile": _P2_VIDEO_PROFILES["animatediff-video-to-video"],
            "capability": _animatediff_extended_capability(
                "AnimateDiffVideoToVideoPipeline",
                "video_to_video",
                required_videos=("sourceVideo",),
            ),
            "roles": _V2V_GRAPH_ROLES,
            "edges": _V2V_GRAPH_EDGES,
            "bindings": _ANIMATEDIFF_V2V_GRAPH_BINDINGS,
        },
        "animatediff-controlnet:control-to-video:v1": {
            "modelType": "AnimateDiffControlNetPipeline",
            "mode": "control_to_video",
            "profile": _P2_VIDEO_PROFILES["animatediff-controlnet"],
            "capability": _animatediff_extended_capability(
                "AnimateDiffControlNetPipeline",
                "control_to_video",
                required_videos=("controlVideo",),
                controlnet=True,
            ),
            "roles": _ANIMATEDIFF_CONTROL_GRAPH_ROLES,
            "edges": _ANIMATEDIFF_CONTROL_GRAPH_EDGES,
            "bindings": _ANIMATEDIFF_CONTROL_GRAPH_BINDINGS,
        },
        "animatediff-controlnet-video-to-video:control-video-to-video:v1": {
            "modelType": "AnimateDiffVideoToVideoControlNetPipeline",
            "mode": "control_video_to_video",
            "profile": _P2_VIDEO_PROFILES["animatediff-controlnet-video-to-video"],
            "capability": _animatediff_extended_capability(
                "AnimateDiffVideoToVideoControlNetPipeline",
                "control_video_to_video",
                required_videos=("sourceVideo", "controlVideo"),
                controlnet=True,
            ),
            "roles": _ANIMATEDIFF_CONTROL_V2V_GRAPH_ROLES,
            "edges": _ANIMATEDIFF_CONTROL_V2V_GRAPH_EDGES,
            "bindings": _ANIMATEDIFF_CONTROL_V2V_GRAPH_BINDINGS,
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
        "cogvideox-2b-video-to-video:video-to-video:v1": {
            "modelType": "CogVideoXVideoToVideoPipeline",
            "mode": "video_to_video",
            "profile": _P2_VIDEO_PROFILES["cogvideox-2b-video-to-video"],
            "capability": _cogvideox_v2v_capability(),
            "roles": _V2V_GRAPH_ROLES,
            "edges": _V2V_GRAPH_EDGES,
            "bindings": _COGVIDEOX_V2V_GRAPH_BINDINGS,
        },
        "allegro:text-to-video:v1": {
            "modelType": "AllegroPipeline",
            "mode": "text_to_video",
            "profile": _P2_VIDEO_PROFILES["allegro"],
            "capability": _allegro_capability(),
            "roles": _VIDEO_GRAPH_ROLES,
            "edges": _VIDEO_GRAPH_EDGES,
            "bindings": _COGVIDEOX_GRAPH_BINDINGS,
        },
        "latte:text-to-video:v1": {
            "modelType": "LattePipeline",
            "mode": "text_to_video",
            "profile": _P2_VIDEO_PROFILES["latte"],
            "capability": _latte_capability(),
            "roles": _VIDEO_GRAPH_ROLES,
            "edges": _VIDEO_GRAPH_EDGES,
            "bindings": _COGVIDEOX_GRAPH_BINDINGS,
        },
        "mochi:text-to-video:v1": {
            "modelType": "MochiPipeline",
            "mode": "text_to_video",
            "profile": _P2_VIDEO_PROFILES["mochi"],
            "capability": _mochi_capability(),
            "roles": _VIDEO_GRAPH_ROLES,
            "edges": _VIDEO_GRAPH_EDGES,
            "bindings": _COGVIDEOX_GRAPH_BINDINGS,
        },
        "sana-video-480p:text-to-video:v1": {
            "modelType": "SanaVideoPipeline",
            "mode": "text_to_video",
            "profile": _P2_VIDEO_PROFILES["sana-video-480p"],
            "capability": _sana_video_capability(image_conditioned=False),
            "roles": _VIDEO_GRAPH_ROLES,
            "edges": _VIDEO_GRAPH_EDGES,
            "bindings": _COGVIDEOX_GRAPH_BINDINGS,
        },
        "sana-video-480p:image-to-video:v1": {
            "modelType": "SanaImageToVideoPipeline",
            "mode": "image_to_video",
            "profile": _P2_VIDEO_PROFILES["sana-video-480p-i2v"],
            "capability": _sana_video_capability(image_conditioned=True),
            "roles": _I2V_GRAPH_ROLES,
            "edges": _I2V_GRAPH_EDGES,
            "bindings": _SANA_VIDEO_I2V_GRAPH_BINDINGS,
        },
        "wan-1.3b:modular-text-to-video:v1": {
            "modelType": "WanModularPipeline",
            "mode": "text_to_video",
            "profile": _P2_VIDEO_PROFILES["wan-1.3b-modular"],
            "capability": _WAN_MODULAR_TEXT_CAPABILITY,
            "roles": _WAN_TEXT_TO_VIDEO_GRAPH_ROLES,
            "edges": _WAN_TEXT_TO_VIDEO_GRAPH_EDGES,
            "bindings": _WAN_TEXT_TO_VIDEO_GRAPH_BINDINGS,
            "autoRequirements": _WAN_MODULAR_TEXT_AUTO_REQUIREMENTS,
        },
        "wan-i2v-480p:modular-image-to-video:v1": {
            "modelType": "WanImage2VideoModularPipeline",
            "mode": "single_image_to_video",
            "profile": _P2_VIDEO_PROFILES["wan-i2v-480p-modular"],
            "capability": _WAN_MODULAR_I2V_CAPABILITY,
            "roles": _WAN_IMAGE_TO_VIDEO_GRAPH_ROLES,
            "edges": _WAN_IMAGE_TO_VIDEO_GRAPH_EDGES,
            "bindings": _WAN_IMAGE_TO_VIDEO_GRAPH_BINDINGS,
            "autoRequirements": _WAN_MODULAR_I2V_AUTO_REQUIREMENTS,
        },
        "wan-flf:image-to-video:v1": {
            "modelType": "WanImage2VideoModularPipeline",
            "mode": "image_to_video",
            "profile": _P2_VIDEO_PROFILES["wan-flf"],
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
    download_files: list[str],
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
        "downloadFiles": download_files,
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
        DDPM_CIFAR10_DIFFUSERS_FILES,
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
        DDPM_CIFAR10_DIFFUSERS_FILES,
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
        CONSISTENCY_IMAGENET64_DIFFUSERS_FILES,
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
                download_files,
                side=side,
                steps=steps,
            ),
            "roles": _UNCONDITIONAL_GRAPH_ROLES,
            "edges": _UNCONDITIONAL_GRAPH_EDGES,
            "bindings": _UNCONDITIONAL_GRAPH_BINDINGS,
        }
        for spec_id, profile_id, pipeline_class, label, family, repo, download_files, side, steps in _P3_UNCONDITIONAL_DEFINITIONS
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
    "downloadFiles": SD15_SHARED_DIFFUSERS_FILES,
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
        "modes": [
            "text_to_image",
            "edit_image",
            "inpaint",
            "control_image",
            "control_edit_image",
            "control_inpaint",
        ],
        "modeRequirements": {
            **_SD15_CONTROLNET_CAPABILITY["modeRequirements"],
            "control_image": {
                "modelRequirements": studio_model_requirements_for_pair("StableDiffusionPipeline", "control_image"),
                "requiredImages": ["controlImage"],
                "note": "Requires one control image and the immutable Canny ControlNet component.",
            },
            "control_edit_image": {
                "modelRequirements": studio_model_requirements_for_pair(
                    "StableDiffusionPipeline", "control_edit_image"
                ),
                "requiredImages": ["referenceImages", "controlImage"],
                "note": "Requires one source image, one control image, and the immutable Canny ControlNet component.",
            },
            "control_inpaint": {
                "modelRequirements": studio_model_requirements_for_pair("StableDiffusionPipeline", "control_inpaint"),
                "requiredImages": ["referenceImages", "maskImage", "controlImage"],
                "note": (
                    "Requires one source image, one mask, one control image, "
                    "and the immutable Canny ControlNet component."
                ),
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
    STUDIO_EXECUTION_SPEC_DEFINITIONS[_sd15_spec_id]["capability"] = deepcopy(_SD15_CONTROLNET_CAPABILITY)
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
_SD15_CONTROLNET_IMG2IMG_PROFILE = _sd15_profile(
    "sd15-controlnet-canny:img2img-direct",
    "control_edit_image",
    "StableDiffusionControlNetImg2ImgPipeline",
)
_SD15_CONTROLNET_IMG2IMG_PROFILE["live_proof"] = False
STUDIO_EXECUTION_SPEC_DEFINITIONS["sd15-controlnet-canny:control-edit-image:v1"] = {
    "modelType": "StableDiffusionPipeline",
    "mode": "control_edit_image",
    "profile": _SD15_CONTROLNET_IMG2IMG_PROFILE,
    "capability": _SD15_CONTROLNET_CAPABILITY,
    "roles": _CONDITIONED_CONTROL_EDIT_GRAPH_ROLES,
    "edges": _CONDITIONED_CONTROL_EDIT_GRAPH_EDGES,
    "bindings": _CONDITIONED_CONTROL_EDIT_GRAPH_BINDINGS,
}
_SD15_CONTROLNET_INPAINT_PROFILE = _sd15_profile(
    "sd15-controlnet-canny:inpaint-direct",
    "control_inpaint",
    "StableDiffusionControlNetInpaintPipeline",
)
_SD15_CONTROLNET_INPAINT_PROFILE["live_proof"] = False
STUDIO_EXECUTION_SPEC_DEFINITIONS["sd15-controlnet-canny:control-inpaint:v1"] = {
    "modelType": "StableDiffusionPipeline",
    "mode": "control_inpaint",
    "profile": _SD15_CONTROLNET_INPAINT_PROFILE,
    "capability": _SD15_CONTROLNET_CAPABILITY,
    "roles": _CONDITIONED_CONTROL_INPAINT_GRAPH_ROLES,
    "edges": _CONDITIONED_CONTROL_INPAINT_GRAPH_EDGES,
    "bindings": _CONDITIONED_CONTROL_INPAINT_GRAPH_BINDINGS,
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
    "downloadFiles": SDXL_TURBO_FP16_DIFFUSERS_FILES,
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
    "revisionCandidates": [require_catalog_revision(SDXL_TURBO_REPO, model_type="StableDiffusionXLTurboPipeline")],
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
    "downloadFiles": SDXL_INSTRUCT_PIX2PIX_DIFFUSERS_FILES,
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
    "downloadFiles": SDXL_BASE_FP16_DIFFUSERS_FILES,
    "artifactLabel": "Diffusers fp16 safetensors assembly",
    "defaultDtype": "float16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 50,
    "recommendedGuidance": 5.0,
    "guidanceLabel": "Guidance",
    "conditioningScale": 0.5,
    "supportsImageInput": True,
    "supportsMask": True,
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
    "modes": ["control_image", "control_edit_image", "control_inpaint"],
    "modeRequirements": {
        "control_image": {
            "modelRequirements": studio_model_requirements_for_pair(
                "StableDiffusionXLControlNetPipeline", "control_image"
            ),
            "requiredImages": ["controlImage"],
            "note": "Requires one control image and the immutable SDXL Canny ControlNet component.",
        },
        "control_edit_image": {
            "modelRequirements": studio_model_requirements_for_pair(
                "StableDiffusionXLControlNetPipeline", "control_edit_image"
            ),
            "requiredImages": ["referenceImages", "controlImage"],
            "note": "Requires one source image, one control image, and the immutable SDXL Canny ControlNet component.",
        },
        "control_inpaint": {
            "modelRequirements": studio_model_requirements_for_pair(
                "StableDiffusionXLControlNetPipeline", "control_inpaint"
            ),
            "requiredImages": ["referenceImages", "maskImage", "controlImage"],
            "note": (
                "Requires one source image, one mask, one control image, "
                "and the immutable SDXL Canny ControlNet component."
            ),
        },
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(SDXL_BASE_REPO, model_type="StableDiffusionXLControlNetPipeline")],
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
_SDXL_CONTROLNET_IMG2IMG_PROFILE = {
    **_SDXL_CONTROLNET_PROFILE,
    "id": "sdxl-controlnet-canny:img2img-direct",
    "modes": ("control_edit_image",),
    "pipeline_class": "StableDiffusionXLControlNetImg2ImgPipeline",
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["sdxl-controlnet-canny:control-edit-image:v1"] = {
    "modelType": "StableDiffusionXLControlNetPipeline",
    "mode": "control_edit_image",
    "profile": _SDXL_CONTROLNET_IMG2IMG_PROFILE,
    "capability": _SDXL_CONTROLNET_CAPABILITY,
    "roles": _CONDITIONED_CONTROL_EDIT_GRAPH_ROLES,
    "edges": _CONDITIONED_CONTROL_EDIT_GRAPH_EDGES,
    "bindings": _CONDITIONED_CONTROL_EDIT_GRAPH_BINDINGS,
}
_SDXL_CONTROLNET_INPAINT_PROFILE = {
    **_SDXL_CONTROLNET_PROFILE,
    "id": "sdxl-controlnet-canny:inpaint-direct",
    "modes": ("control_inpaint",),
    "pipeline_class": "StableDiffusionXLControlNetInpaintPipeline",
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["sdxl-controlnet-canny:control-inpaint:v1"] = {
    "modelType": "StableDiffusionXLControlNetPipeline",
    "mode": "control_inpaint",
    "profile": _SDXL_CONTROLNET_INPAINT_PROFILE,
    "capability": _SDXL_CONTROLNET_CAPABILITY,
    "roles": _CONDITIONED_CONTROL_INPAINT_GRAPH_ROLES,
    "edges": _CONDITIONED_CONTROL_INPAINT_GRAPH_EDGES,
    "bindings": _CONDITIONED_CONTROL_INPAINT_GRAPH_BINDINGS,
}


_HUNYUAN_DIT_PROFILE = {
    "id": "hunyuan-dit-v1-2-distilled:direct",
    "model_type": "HunyuanDiTPipeline",
    "modes": ("text_to_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "HunyuanDiTPipeline",
    "default_repo": HUNYUAN_DIT_DISTILLED_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 25,
    "live_proof": False,
    "compatible_repos": (),
}
_HUNYUAN_DIT_CAPABILITY = {
    "modelType": "HunyuanDiTPipeline",
    "label": "Hunyuan-DiT",
    "displayName": "Hunyuan-DiT v1.2 Distilled",
    "family": "Hunyuan-DiT",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": HUNYUAN_DIT_DISTILLED_REPO,
    "downloadFiles": HUNYUAN_DIT_DISTILLED_DIFFUSERS_FILES,
    "artifactLabel": "Official Tencent community-license safetensors snapshot",
    "defaultDtype": "float16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 25,
    "recommendedGuidance": 5.0,
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
        "steps": 25,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["text_to_image"],
    "modeRequirements": {},
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(HUNYUAN_DIT_DISTILLED_REPO, model_type="HunyuanDiTPipeline")],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The exact v1.2 distilled snapshot is shared with the separately admitted Canny ControlNet assembly.",
        "The bounded standalone recipe uses 1024x1024, the model card's 25-step distilled path, package-default guidance 5, and at most 256 T5 tokens.",
        "The Tencent community license and acceptable-use obligations require explicit acknowledgement before graph creation or installation.",
        "The immutable model index declares a required safety checker but ships none; Auto and Gallery remain disabled until live safety and quality review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["hunyuan-dit-v1-2-distilled:text-to-image:v1"] = {
    "modelType": "HunyuanDiTPipeline",
    "mode": "text_to_image",
    "profile": _HUNYUAN_DIT_PROFILE,
    "capability": _HUNYUAN_DIT_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _SDXL_GRAPH_BINDINGS,
}
_HUNYUAN_DIT_PAG_PROFILE = {
    **_HUNYUAN_DIT_PROFILE,
    "id": "hunyuan-dit-v1-2-distilled-pag:direct",
    "model_type": "HunyuanDiTPAGPipeline",
    "pipeline_class": "HunyuanDiTPAGPipeline",
    "live_proof": False,
}
_HUNYUAN_DIT_PAG_CAPABILITY = deepcopy(_HUNYUAN_DIT_CAPABILITY)
_HUNYUAN_DIT_PAG_CAPABILITY.update(
    {
        "modelType": "HunyuanDiTPAGPipeline",
        "label": "Hunyuan-DiT PAG",
        "displayName": "Hunyuan-DiT v1.2 Distilled PAG",
        "recommendedPagScale": 3.0,
        "recommendedPagAdaptiveScale": 0.0,
        "recommendedGuidance": 4.0,
        "notes": [
            "Perturbed-attention guidance reuses the immutable Hunyuan-DiT v1.2 distilled safetensors snapshot without an auxiliary artifact.",
            "The exact generic recipe is fixed at 1024x1024, at most 25 steps, guidance 4, PAG scale 3, adaptive scale 0, and official transformer layer 14; the reviewed PAG call fixes both encoder lengths internally.",
            "The Tencent community license and acceptable-use obligations require explicit acknowledgement; Auto and Gallery remain disabled pending live review.",
        ],
    }
)
_HUNYUAN_DIT_PAG_CAPABILITY.pop("recommendedMaxSequenceLength", None)
STUDIO_EXECUTION_SPEC_DEFINITIONS["hunyuan-dit-v1-2-distilled-pag:text-to-image:v1"] = {
    "modelType": "HunyuanDiTPAGPipeline",
    "mode": "text_to_image",
    "profile": _HUNYUAN_DIT_PAG_PROFILE,
    "capability": _HUNYUAN_DIT_PAG_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _HUNYUAN_PAG_GRAPH_BINDINGS,
}


_HUNYUAN_DIT_CONTROLNET_CAPABILITY = {
    "modelType": "HunyuanDiTControlNetPipeline",
    "label": "Hunyuan-DiT v1.2 ControlNet",
    "displayName": "Hunyuan-DiT v1.2 Canny ControlNet",
    "family": "Hunyuan-DiT",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": HUNYUAN_DIT_DISTILLED_REPO,
    "downloadFiles": HUNYUAN_DIT_DISTILLED_DIFFUSERS_FILES,
    "artifactLabel": "Official Tencent community-license safetensors assembly",
    "defaultDtype": "float16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 50,
    "recommendedGuidance": 6.0,
    "recommendedMaxSequenceLength": 256,
    "guidanceLabel": "Guidance",
    "conditioningScale": 1.0,
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
            "modelRequirements": studio_model_requirements_for_pair("HunyuanDiTControlNetPipeline", "control_image"),
            "requiredImages": ["controlImage"],
            "note": "Requires one source image for the canonical Canny preprocessor and exact v1.2 ControlNet.",
        },
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(HUNYUAN_DIT_DISTILLED_REPO, model_type="HunyuanDiTControlNetPipeline")
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The source graph is fixed to the publisher's 1024px, 50-step, guidance-6, scale-1 Canny recipe.",
        "The base and ControlNet are loaded in float16 from exact safetensors snapshots with model CPU offload.",
        "The Tencent community license and acceptable-use obligations require explicit acknowledgement before graph creation or installation.",
        "The immutable base declares a required safety checker but ships none; Auto and Gallery remain disabled until live safety and quality review.",
        "The separately reviewed Depth and Pose components remain optional expert substitutions without canonical workflows.",
    ],
}
_HUNYUAN_DIT_CONTROLNET_PROFILE = {
    "id": "hunyuan-dit-v1-2-controlnet-canny:direct",
    "model_type": "HunyuanDiTControlNetPipeline",
    "modes": ("control_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "HunyuanDiTControlNetPipeline",
    "default_repo": HUNYUAN_DIT_DISTILLED_REPO,
    "fallback_repo": None,
    "quantizable_components": ("transformer", "text_encoder", "text_encoder_2"),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 50,
    "live_proof": False,
    "compatible_repos": (),
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["hunyuan-dit-v1-2-controlnet-canny:control-image:v1"] = {
    "modelType": "HunyuanDiTControlNetPipeline",
    "mode": "control_image",
    "profile": _HUNYUAN_DIT_CONTROLNET_PROFILE,
    "capability": _HUNYUAN_DIT_CONTROLNET_CAPABILITY,
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
    "downloadFiles": SDXL_BASE_FP16_DIFFUSERS_FILES,
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
    "revisionCandidates": [require_catalog_revision(SDXL_BASE_REPO, model_type="StableDiffusionXLAdapterPipeline")],
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
    "downloadFiles": SDXL_BASE_FP16_DIFFUSERS_FILES,
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
    "supportsControlImage": True,
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
    "modes": ["text_to_image", "edit_image", "inpaint", "control_image", "control_edit_image"],
    "modeRequirements": {
        "edit_image": {
            "requiredImages": ["referenceImages"],
            "note": "Requires one source image for PAG image-to-image transformation.",
        },
        "inpaint": {
            "requiredImages": ["referenceImages", "maskImage"],
            "note": "Requires one source image and one mask image for PAG inpainting.",
        },
        "control_image": {
            "modelRequirements": studio_model_requirements_for_pair("StableDiffusionXLPAGPipeline", "control_image"),
            "requiredImages": ["controlImage"],
            "note": "Requires one control image and the immutable SDXL Canny ControlNet component.",
        },
        "control_edit_image": {
            "modelRequirements": studio_model_requirements_for_pair(
                "StableDiffusionXLPAGPipeline", "control_edit_image"
            ),
            "requiredImages": ["referenceImages", "controlImage"],
            "note": "Requires one source image, one control image, and the immutable SDXL Canny ControlNet component.",
        },
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(SDXL_BASE_REPO, model_type="StableDiffusionXLPAGPipeline")],
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
_SDXL_PAG_CONTROLNET_PROFILE = {
    **_SDXL_PAG_PROFILE,
    "id": "sdxl-pag-controlnet-canny:direct",
    "modes": ("control_image",),
    "pipeline_class": "StableDiffusionXLControlNetPAGPipeline",
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["sdxl-pag-controlnet-canny:control-image:v1"] = {
    "modelType": "StableDiffusionXLPAGPipeline",
    "mode": "control_image",
    "profile": _SDXL_PAG_CONTROLNET_PROFILE,
    "capability": _SDXL_PAG_CAPABILITY,
    "roles": _CONDITIONED_CONTROL_GRAPH_ROLES,
    "edges": _CONDITIONED_CONTROL_GRAPH_EDGES,
    "bindings": _PAG_CONDITIONED_CONTROL_GRAPH_BINDINGS,
}
_SDXL_PAG_CONTROLNET_IMG2IMG_PROFILE = {
    **_SDXL_PAG_PROFILE,
    "id": "sdxl-pag-controlnet-canny:img2img-direct",
    "modes": ("control_edit_image",),
    "pipeline_class": "StableDiffusionXLControlNetPAGImg2ImgPipeline",
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["sdxl-pag-controlnet-canny:control-edit-image:v1"] = {
    "modelType": "StableDiffusionXLPAGPipeline",
    "mode": "control_edit_image",
    "profile": _SDXL_PAG_CONTROLNET_IMG2IMG_PROFILE,
    "capability": _SDXL_PAG_CAPABILITY,
    "roles": _CONDITIONED_CONTROL_EDIT_GRAPH_ROLES,
    "edges": _CONDITIONED_CONTROL_EDIT_GRAPH_EDGES,
    "bindings": _PAG_CONDITIONED_CONTROL_EDIT_GRAPH_BINDINGS,
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
    "downloadFiles": SANA_600M_FP16_DIFFUSERS_FILES,
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
_SANA_PAG_PROFILE = {
    **_SANA_PROFILE,
    "id": "sana-600m-pag:direct",
    "model_type": "SanaPAGPipeline",
    "pipeline_class": "SanaPAGPipeline",
    "live_proof": False,
}
_SANA_PAG_CAPABILITY = deepcopy(_SANA_CAPABILITY)
_SANA_PAG_CAPABILITY.update(
    {
        "modelType": "SanaPAGPipeline",
        "label": "Sana 0.6B PAG",
        "displayName": "Sana 0.6B 1024px PAG",
        "recommendedPagScale": 3.0,
        "recommendedPagAdaptiveScale": 0.0,
        "notes": [
            "Perturbed-attention guidance reuses the immutable Sana 0.6B fp16 safetensors snapshot without an auxiliary artifact.",
            "The reviewed generic recipe uses 1024x1024, 20 steps, guidance 4.5, at most 300 prompt tokens, PAG scale 3, and adaptive scale 0.",
            "Apache-2.0 applies alongside the bundled Gemma terms and prohibited-use policy; Auto and Gallery remain disabled pending live review.",
        ],
    }
)
STUDIO_EXECUTION_SPEC_DEFINITIONS["sana-600m-pag:text-to-image:v1"] = {
    "modelType": "SanaPAGPipeline",
    "mode": "text_to_image",
    "profile": _SANA_PAG_PROFILE,
    "capability": _SANA_PAG_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _PAG_TEXT_TO_IMAGE_BINDINGS,
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
    "downloadFiles": SANA_SPRINT_DIFFUSERS_FILES,
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
    "revisionCandidates": [require_catalog_revision(SANA_SPRINT_REPO, model_type="SanaSprintPipeline")],
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
    "downloadFiles": PIXART_SIGMA_DIFFUSERS_FILES,
    "artifactLabel": "OpenRAIL++ Diffusers safetensors repo",
    "defaultDtype": "float32",
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
        "dtype": "float32",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "steps": 20,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["text_to_image"],
    "modeRequirements": {},
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(PIXART_SIGMA_REPO, model_type="PixArtSigmaPipeline")],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The immutable public snapshot contains four safetensors files and uses only package-owned Diffusers and Transformers classes.",
        "The reviewed 1024px recipe uses FP32, 20 steps, guidance 4.5, at most 300 prompt tokens, and explicit model or sequential CPU offload. FP16/BF16 require separate platform qualification because the qualified ROCm APU produced non-finite denoising latents.",
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
_PIXART_SIGMA_PAG_PROFILE = {
    **_PIXART_SIGMA_PROFILE,
    "id": "pixart-sigma-1024-pag:direct",
    "model_type": "PixArtSigmaPAGPipeline",
    "pipeline_class": "PixArtSigmaPAGPipeline",
    "live_proof": False,
}
_PIXART_SIGMA_PAG_CAPABILITY = deepcopy(_PIXART_SIGMA_CAPABILITY)
_PIXART_SIGMA_PAG_CAPABILITY.update(
    {
        "modelType": "PixArtSigmaPAGPipeline",
        "label": "PixArt Sigma PAG",
        "displayName": "PixArt Sigma XL 1024px PAG",
        "recommendedGuidance": 1.0,
        "recommendedPagScale": 4.0,
        "recommendedPagAdaptiveScale": 0.0,
        "notes": [
            "Perturbed-attention guidance reuses the immutable PixArt Sigma XL 1024px safetensors snapshot without an auxiliary artifact and applies PAG at the official block 14 surface.",
            "The official PixArt PAG recipe uses 1024x1024, 20 steps, guidance 1, at most 300 prompt tokens, PAG scale 4, and adaptive scale 0.",
            "The approximately 21.83 GB selected weight surface is remote-only; Auto and Gallery remain disabled pending live output review.",
        ],
    }
)
STUDIO_EXECUTION_SPEC_DEFINITIONS["pixart-sigma-1024-pag:text-to-image:v1"] = {
    "modelType": "PixArtSigmaPAGPipeline",
    "mode": "text_to_image",
    "profile": _PIXART_SIGMA_PAG_PROFILE,
    "capability": _PIXART_SIGMA_PAG_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _PAG_TEXT_TO_IMAGE_BINDINGS,
}


_KANDINSKY3_PROFILE = {
    "id": "kandinsky3:direct",
    "model_type": "Kandinsky3Pipeline",
    "modes": ("text_to_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "Kandinsky3Pipeline",
    "default_repo": KANDINSKY3_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 25,
    "live_proof": False,
    "compatible_repos": (),
}
_KANDINSKY3_CAPABILITY = {
    "modelType": "Kandinsky3Pipeline",
    "label": "Kandinsky 3",
    "displayName": "Kandinsky 3",
    "family": "Kandinsky",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": KANDINSKY3_REPO,
    "downloadFiles": KANDINSKY3_FP16_DIFFUSERS_FILES,
    "artifactLabel": "Apache-2.0-declared fp16 Diffusers safetensors repo",
    "defaultDtype": "float16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 25,
    "recommendedGuidance": 3.0,
    "recommendedMaxSequenceLength": 128,
    "guidanceLabel": "Guidance",
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
        "dtype": "float16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "steps": 25,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["text_to_image", "edit_image"],
    "modeRequirements": {"edit_image": {"requiredImages": ["referenceImages"]}},
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(KANDINSKY3_REPO, model_type="Kandinsky3Pipeline")],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The immutable public snapshot contains only package-owned Diffusers and Transformers classes and seven fp16 safetensors files.",
        "The reviewed single-stage routes use the package's 1024px default, 25 steps, guidance 3, at most 128 prompt tokens, and explicit model or sequential CPU offload.",
        "The approximately 28.39 GB selected weight surface has no safety checker and remains remote-only; Auto and Gallery are disabled pending live output review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["kandinsky3:text-to-image:v1"] = {
    "modelType": "Kandinsky3Pipeline",
    "mode": "text_to_image",
    "profile": _KANDINSKY3_PROFILE,
    "capability": _KANDINSKY3_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _SDXL_GRAPH_BINDINGS,
}
_KANDINSKY3_IMG2IMG_PROFILE = {
    **_KANDINSKY3_PROFILE,
    "id": "kandinsky3:img2img-direct",
    "modes": ("edit_image",),
    "pipeline_class": "Kandinsky3Img2ImgPipeline",
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["kandinsky3:edit-image:v1"] = {
    "modelType": "Kandinsky3Pipeline",
    "mode": "edit_image",
    "profile": _KANDINSKY3_IMG2IMG_PROFILE,
    "capability": _KANDINSKY3_CAPABILITY,
    "roles": _EDIT_GRAPH_ROLES,
    "edges": _EDIT_GRAPH_EDGES,
    "bindings": _SDXL_EDIT_GRAPH_BINDINGS,
}


_LONGCAT_IMAGE_PROFILE = {
    "id": "longcat-image:direct",
    "model_type": "LongCatImagePipeline",
    "modes": ("text_to_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "LongCatImagePipeline",
    "default_repo": LONGCAT_IMAGE_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 50,
    "live_proof": False,
    "compatible_repos": (),
}
_LONGCAT_IMAGE_CAPABILITY = {
    "modelType": "LongCatImagePipeline",
    "label": "LongCat Image",
    "displayName": "LongCat Image 6B",
    "family": "LongCat Image",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": LONGCAT_IMAGE_REPO,
    "downloadFiles": LONGCAT_IMAGE_DIFFUSERS_FILES,
    "artifactLabel": "Apache-2.0-declared bfloat16 Diffusers safetensors repo",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 50,
    "recommendedGuidance": 4.0,
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
        "steps": 50,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["text_to_image"],
    "modeRequirements": {},
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(LONGCAT_IMAGE_REPO, model_type="LongCatImagePipeline")],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The immutable public snapshot contains seven exact safetensors files and executes only package-owned Diffusers and Transformers classes.",
        "MoDiff bounds generation to 512-2048px sides, at most 1,048,576 output pixels, 50 steps, guidance 4, and disables the package's autoregressive prompt rewrite.",
        "The approximately 29.29 GB selected weight surface has no safety checker and remains remote-only; Auto and Gallery are disabled pending live output review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["longcat-image:text-to-image:v1"] = {
    "modelType": "LongCatImagePipeline",
    "mode": "text_to_image",
    "profile": _LONGCAT_IMAGE_PROFILE,
    "capability": _LONGCAT_IMAGE_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _SDXL_GRAPH_BINDINGS,
}

_LONGCAT_IMAGE_EDIT_PROFILE = {
    **_LONGCAT_IMAGE_PROFILE,
    "id": "longcat-image-edit:direct",
    "model_type": "LongCatImageEditPipeline",
    "modes": ("edit_image",),
    "pipeline_class": "LongCatImageEditPipeline",
    "default_repo": LONGCAT_IMAGE_EDIT_REPO,
}
_LONGCAT_IMAGE_EDIT_CAPABILITY = {
    **_LONGCAT_IMAGE_CAPABILITY,
    "modelType": "LongCatImageEditPipeline",
    "label": "LongCat Image Edit",
    "displayName": "LongCat Image Edit 6B",
    "defaultRepo": LONGCAT_IMAGE_EDIT_REPO,
    "downloadFiles": LONGCAT_IMAGE_EDIT_DIFFUSERS_FILES,
    "recommendedGuidance": 4.5,
    "supportsImageInput": True,
    "modes": ["edit_image"],
    "modeRequirements": {
        "edit_image": {
            "requiredImages": ["referenceImages"],
            "note": "Requires one source image between 1:4 and 4:1 aspect ratio; output is normalized to an approximately one-megapixel bucket.",
        }
    },
    "revisionCandidates": [require_catalog_revision(LONGCAT_IMAGE_EDIT_REPO, model_type="LongCatImageEditPipeline")],
    "notes": [
        "The immutable public edit snapshot shares the exact text encoder and VAE identities with the generation model and replaces only the safetensors transformer.",
        "MoDiff accepts exactly one at-most-1,048,576-pixel source between 1:4 and 4:1 aspect ratio; the package derives an approximately one-megapixel output and runs at most 50 steps.",
        "The approximately 29.29 GB selected weight surface has no safety checker and remains remote-only; Auto and Gallery are disabled pending live output review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["longcat-image-edit:edit-image:v1"] = {
    "modelType": "LongCatImageEditPipeline",
    "mode": "edit_image",
    "profile": _LONGCAT_IMAGE_EDIT_PROFILE,
    "capability": _LONGCAT_IMAGE_EDIT_CAPABILITY,
    "roles": _EDIT_GRAPH_ROLES,
    "edges": _EDIT_GRAPH_EDGES,
    "bindings": _SDXL_EDIT_GRAPH_BINDINGS,
}


_LUMINA_PROFILE = {
    "id": "lumina-next:direct",
    "model_type": "LuminaPipeline",
    "modes": ("text_to_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "LuminaPipeline",
    "default_repo": LUMINA_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 30,
    "live_proof": False,
    "compatible_repos": (),
}
_LUMINA_CAPABILITY = {
    "modelType": "LuminaPipeline",
    "label": "Lumina Next",
    "displayName": "Lumina Next SFT 2B",
    "family": "Lumina",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": LUMINA_REPO,
    "downloadFiles": LUMINA_NEXT_DIFFUSERS_FILES,
    "artifactLabel": "Apache-2.0-declared bfloat16 Diffusers safetensors repo",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 30,
    "recommendedGuidance": 4.0,
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
        "dtype": "bfloat16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "steps": 30,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["text_to_image"],
    "modeRequirements": {},
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(LUMINA_REPO, model_type="LuminaPipeline")],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The immutable public snapshot contains four exact safetensors weight files and executes only package-owned Diffusers and Transformers classes.",
        "MoDiff bounds generation to 512-2048px sides, at most 1,048,576 output pixels, 30 steps, guidance 4, 256 prompt tokens, and disables optional caption cleaning.",
        "The approximately 8.86 GB weight surface has no safety checker and remains remote-only; Auto and Gallery are disabled pending live output review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["lumina-next:text-to-image:v1"] = {
    "modelType": "LuminaPipeline",
    "mode": "text_to_image",
    "profile": _LUMINA_PROFILE,
    "capability": _LUMINA_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _SDXL_GRAPH_BINDINGS,
}

_LUMINA2_PROFILE = {
    **_LUMINA_PROFILE,
    "id": "lumina2:direct",
    "model_type": "Lumina2Pipeline",
    "pipeline_class": "Lumina2Pipeline",
    "default_repo": LUMINA2_REPO,
    "max_low_memory_steps": 50,
}
_LUMINA2_CAPABILITY = {
    **_LUMINA_CAPABILITY,
    "modelType": "Lumina2Pipeline",
    "label": "Lumina Image 2.0",
    "displayName": "Lumina Image 2.0 2.6B",
    "family": "Lumina 2.0",
    "defaultRepo": LUMINA2_REPO,
    "artifactLabel": "Apache-2.0-declared Diffusers safetensors-only selection",
    "downloadFiles": LUMINA2_DIFFUSERS_FILES,
    "recommendedSteps": 50,
    "revisionCandidates": [require_catalog_revision(LUMINA2_REPO, model_type="Lumina2Pipeline")],
    "lowVram": {
        **_LUMINA_CAPABILITY["lowVram"],
        "steps": 50,
    },
    "notes": [
        "The immutable public snapshot contains six exact safetensors weight files; MoDiff's 17-file download allowlist excludes the legacy pickle checkpoints and demo asset.",
        "MoDiff bounds generation to 512-2048px sides, at most 1,048,576 output pixels, 50 steps, guidance 4, 256 prompt tokens, and fixes the official CFG truncation ratio to 0.25 with normalization enabled.",
        "The approximately 21.23 GB safetensors surface has no safety checker and remains remote-only; Auto and Gallery are disabled pending live output review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["lumina2:text-to-image:v1"] = {
    "modelType": "Lumina2Pipeline",
    "mode": "text_to_image",
    "profile": _LUMINA2_PROFILE,
    "capability": _LUMINA2_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _SDXL_GRAPH_BINDINGS,
}


_OMNIGEN_PROFILE = {
    "id": "omnigen-v1:direct",
    "model_type": "OmniGenPipeline",
    "modes": ("text_to_image", "edit_image", "multi_image_reference_edit"),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "OmniGenPipeline",
    "default_repo": OMNIGEN_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 50,
    "live_proof": False,
    "compatible_repos": (),
}
_OMNIGEN_CAPABILITY = {
    "modelType": "OmniGenPipeline",
    "label": "OmniGen v1",
    "displayName": "OmniGen v1",
    "family": "OmniGen",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": OMNIGEN_REPO,
    "downloadFiles": OMNIGEN_DIFFUSERS_FILES,
    "artifactLabel": "MIT bfloat16 Diffusers safetensors repo",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 50,
    "recommendedGuidance": 2.5,
    "recommendedMaxSequenceLength": 256,
    "guidanceLabel": "Text guidance",
    "conditioningScale": 1.6,
    "supportsNegativePrompt": False,
    "supportsImageInput": True,
    "supportsMask": False,
    "supportsMultiImage": True,
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
        "steps": 50,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["text_to_image", "edit_image", "multi_image_reference_edit"],
    "modeRequirements": {
        "edit_image": {
            "requiredImages": ["referenceImages"],
            "note": "Requires one source image; MoDiff inserts the package's reserved positional image tag.",
        },
        "multi_image_reference_edit": {
            "requiredImages": ["referenceImages"],
            "note": "Requires 1 to 3 source images; MoDiff inserts one ordered positional tag per reference.",
        },
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(OMNIGEN_REPO, model_type="OmniGenPipeline")],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The immutable public snapshot contains two exact safetensors files and executes only package-owned Diffusers and Transformers classes.",
        "MoDiff bounds output to 512-2048px sides and at most 1,048,576 pixels, input references to three cumulative megapixels with a 1024px preprocessing side, 50 steps, text guidance 2.5, and image guidance 1.6.",
        "Reference placeholders are generated by the backend and user-supplied reserved placeholder syntax is rejected.",
        "The approximately 8.09 GB weight surface has no safety checker and remains remote-only; Auto and Gallery are disabled pending live output review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["omnigen-v1:text-to-image:v1"] = {
    "modelType": "OmniGenPipeline",
    "mode": "text_to_image",
    "profile": _OMNIGEN_PROFILE,
    "capability": _OMNIGEN_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _OMNIGEN_GRAPH_BINDINGS,
}
for _spec_id, _mode in (
    ("omnigen-v1:edit-image:v1", "edit_image"),
    ("omnigen-v1:multi-image-reference-edit:v1", "multi_image_reference_edit"),
):
    STUDIO_EXECUTION_SPEC_DEFINITIONS[_spec_id] = {
        "modelType": "OmniGenPipeline",
        "mode": _mode,
        "profile": _OMNIGEN_PROFILE,
        "capability": _OMNIGEN_CAPABILITY,
        "roles": _EDIT_GRAPH_ROLES,
        "edges": _EDIT_GRAPH_EDGES,
        "bindings": _OMNIGEN_EDIT_GRAPH_BINDINGS,
    }


_OVIS_IMAGE_PROFILE = {
    "id": "ovis-image-7b:direct",
    "model_type": "OvisImagePipeline",
    "modes": ("text_to_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "OvisImagePipeline",
    "default_repo": OVIS_IMAGE_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 50,
    "live_proof": False,
    "compatible_repos": (),
}
_OVIS_IMAGE_CAPABILITY = {
    "modelType": "OvisImagePipeline",
    "label": "Ovis Image",
    "displayName": "Ovis Image 7B",
    "family": "Ovis Image",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": OVIS_IMAGE_REPO,
    "artifactLabel": "Apache-2.0 bfloat16 Diffusers safetensors-only selection",
    "downloadFiles": OVIS_IMAGE_DIFFUSERS_FILES,
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 50,
    "recommendedGuidance": 5.0,
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
        "dtype": "bfloat16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "steps": 50,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["text_to_image"],
    "modeRequirements": {},
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(OVIS_IMAGE_REPO, model_type="OvisImagePipeline")],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The immutable public snapshot executes the package-owned Ovis Image pipeline from a 21-file Diffusers-only allowlist containing five exact safetensors weight files.",
        "The allowlist excludes duplicate root-native checkpoints and the bundled Ovis2.5 repository, including both Python source files, so remote code is never selected or trusted.",
        "MoDiff bounds generation to 512-2048px sides, at most 1,048,576 output pixels, 50 steps, guidance 5, and 256 prompt tokens.",
        "The approximately 21.79 GB selected weight surface has no safety checker and remains remote-only; Auto and Gallery are disabled pending live output review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["ovis-image-7b:text-to-image:v1"] = {
    "modelType": "OvisImagePipeline",
    "mode": "text_to_image",
    "profile": _OVIS_IMAGE_PROFILE,
    "capability": _OVIS_IMAGE_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _SDXL_GRAPH_BINDINGS,
}


_PRX_PROFILE = {
    "id": "prx-512-sft:direct",
    "model_type": "PRXPipeline",
    "modes": ("text_to_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "PRXPipeline",
    "default_repo": PRX_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 512,
    "max_low_memory_steps": 28,
    "live_proof": False,
    "compatible_repos": (),
}
_PRX_CAPABILITY = {
    "modelType": "PRXPipeline",
    "label": "PRX",
    "displayName": "PRX 512 SFT",
    "family": "PRX",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": PRX_REPO,
    "artifactLabel": "Apache-2.0 and Gemma-terms bfloat16 Diffusers safetensors snapshot",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 512, "height": 512, "aspectRatio": "1:1"},
    "recommendedSteps": 28,
    "recommendedGuidance": 5.0,
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
        "dtype": "bfloat16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_SEQUENTIAL_CPU,
        "steps": 28,
        "width": 512,
        "height": 512,
    },
    "modes": ["text_to_image"],
    "modeRequirements": {},
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(PRX_REPO, model_type="PRXPipeline")],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "downloadFiles": PRX_DIFFUSERS_FILES,
    "notes": [
        "The immutable public snapshot contains five exact safetensors weight files, no repository Python, and package-owned PRX execution through the reviewed Diffusers revision.",
        "MoDiff uses the native 512px SFT recipe and bounds requested sides to the package's 352-704px aspect bins, at most 262,144 output pixels, 28 steps, guidance 5, and 256 prompt tokens.",
        "The bundled NOTICE applies Apache-2.0 alongside the incorporated T5-Gemma terms and prohibited-use policy.",
        "The approximately 15.48 GB weight surface has no safety checker and remains remote-only; Auto and Gallery are disabled pending live output review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["prx-512-sft:text-to-image:v1"] = {
    "modelType": "PRXPipeline",
    "mode": "text_to_image",
    "profile": _PRX_PROFILE,
    "capability": _PRX_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _SDXL_GRAPH_BINDINGS,
}


_NUCLEUS_IMAGE_PROFILE = {
    "id": "nucleus-image:direct",
    "model_type": "NucleusMoEImagePipeline",
    "modes": ("text_to_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "NucleusMoEImagePipeline",
    "default_repo": NUCLEUS_IMAGE_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 50,
    "live_proof": False,
    "compatible_repos": (),
}
_NUCLEUS_IMAGE_CAPABILITY = {
    "modelType": "NucleusMoEImagePipeline",
    "label": "Nucleus Image",
    "displayName": "Nucleus Image 17B MoE",
    "family": "Nucleus Image",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": NUCLEUS_IMAGE_REPO,
    "artifactLabel": "Apache-2.0-declared bfloat16 Diffusers safetensors snapshot",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 50,
    "recommendedGuidance": 4.0,
    "recommendedMaxSequenceLength": 1024,
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
        "steps": 50,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["text_to_image"],
    "modeRequirements": {},
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(NUCLEUS_IMAGE_REPO, model_type="NucleusMoEImagePipeline")],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "downloadFiles": NUCLEUS_IMAGE_DIFFUSERS_FILES,
    "notes": [
        "The immutable public snapshot contains 38 selected files, 12 exact safetensors weight files totaling approximately 51.63 GB, no repository Python, and package-owned Nucleus Image execution through the reviewed Diffusers revision.",
        "MoDiff admits the seven official aspect buckets through 768-1344px sides in 32px steps, at most 1,060,864 output pixels, 50 steps, guidance 4, and 1024 prompt tokens.",
        "The model card declares Apache-2.0, but the immutable snapshot contains no license file; clarification remains pending before broader exposure.",
        "This base checkpoint has no post-training or safety checker and remains remote-only; Auto and Gallery are disabled pending live output and policy review.",
        "The app storage preflight did not fit this snapshot beside the existing queue, so it was not submitted and no older model was deleted.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["nucleus-image:text-to-image:v1"] = {
    "modelType": "NucleusMoEImagePipeline",
    "mode": "text_to_image",
    "profile": _NUCLEUS_IMAGE_PROFILE,
    "capability": _NUCLEUS_IMAGE_CAPABILITY,
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
    "downloadFiles": AURAFLOW_V03_DIFFUSERS_FILES,
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
    "revisionCandidates": [require_catalog_revision(AURAFLOW_V03_REPO, model_type="AuraFlowPipeline")],
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
    "downloadFiles": CHROMA1_HD_DIFFUSERS_FILES,
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
    "revisionCandidates": [require_catalog_revision(CHROMA1_HD_REPO, model_type="ChromaPipeline")],
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


_COGVIEW3_PLUS_PROFILE = {
    "id": "cogview3-plus-3b:direct",
    "model_type": "CogView3PlusPipeline",
    "modes": ("text_to_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "CogView3PlusPipeline",
    "default_repo": COGVIEW3_PLUS_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 50,
    "live_proof": False,
    "compatible_repos": (),
}
_COGVIEW3_PLUS_CAPABILITY = {
    "modelType": "CogView3PlusPipeline",
    "label": "CogView3",
    "displayName": "CogView3 Plus 3B",
    "family": "CogView3",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": COGVIEW3_PLUS_REPO,
    "downloadFiles": COGVIEW3_PLUS_DIFFUSERS_FILES,
    "artifactLabel": "Apache-2.0 bfloat16 Diffusers safetensors repo",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 50,
    "recommendedGuidance": 7.0,
    "recommendedMaxSequenceLength": 224,
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
        "steps": 50,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["text_to_image"],
    "modeRequirements": {},
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(COGVIEW3_PLUS_REPO, model_type="CogView3PlusPipeline")],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The immutable public snapshot uses only package-owned Diffusers and Transformers classes and seven bfloat16 safetensors files.",
        "The reviewed route supports 512-to-2048px sides in 32px increments, 50 steps, guidance 7, at most 224 prompt tokens, and VAE slicing and tiling.",
        "The approximately 25.56 GB weight surface is remote-only; Auto and Gallery remain disabled pending live output review.",
        "The immutable model card links a nonexistent LICENSE.md, so the Apache-2.0 identifier is retained as model-card metadata evidence rather than a bundled license-file claim.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["cogview3-plus-3b:text-to-image:v1"] = {
    "modelType": "CogView3PlusPipeline",
    "mode": "text_to_image",
    "profile": _COGVIEW3_PLUS_PROFILE,
    "capability": _COGVIEW3_PLUS_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _SDXL_GRAPH_BINDINGS,
}


_COGVIEW4_6B_PROFILE = {
    "id": "cogview4-6b:direct",
    "model_type": "CogView4Pipeline",
    "modes": ("text_to_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "CogView4Pipeline",
    "default_repo": COGVIEW4_6B_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 50,
    "live_proof": False,
    "compatible_repos": (),
}
_COGVIEW4_6B_CAPABILITY = {
    "modelType": "CogView4Pipeline",
    "label": "CogView4",
    "displayName": "CogView4 6B",
    "family": "CogView4",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": COGVIEW4_6B_REPO,
    "downloadFiles": COGVIEW4_6B_DIFFUSERS_FILES,
    "artifactLabel": "Apache-2.0 bfloat16 Diffusers safetensors repo",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 50,
    "recommendedGuidance": 3.5,
    "recommendedMaxSequenceLength": 1024,
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
        "steps": 50,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["text_to_image"],
    "modeRequirements": {},
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(COGVIEW4_6B_REPO, model_type="CogView4Pipeline")],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The immutable public Apache-2.0 snapshot uses only package-owned Diffusers and Transformers classes and eight bfloat16 safetensors files.",
        "The reviewed route supports 512-to-2048px sides in 32px increments, at most 2^21 output pixels, 50 steps, guidance 3.5, and at most 1024 prompt tokens.",
        "The approximately 31.11 GB weight surface is remote-only; Auto and Gallery remain disabled pending live output review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["cogview4-6b:text-to-image:v1"] = {
    "modelType": "CogView4Pipeline",
    "mode": "text_to_image",
    "profile": _COGVIEW4_6B_PROFILE,
    "capability": _COGVIEW4_6B_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _SDXL_GRAPH_BINDINGS,
}


_ERNIE_IMAGE_TURBO_PROFILE = {
    "id": "ernie-image-turbo:direct",
    "model_type": "ErnieImagePipeline",
    "modes": ("text_to_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "ErnieImagePipeline",
    "default_repo": ERNIE_IMAGE_TURBO_REPO,
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
_ERNIE_IMAGE_TURBO_CAPABILITY = {
    "modelType": "ErnieImagePipeline",
    "label": "ERNIE Image Turbo",
    "displayName": "ERNIE Image Turbo 1024px",
    "family": "ERNIE Image",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": ERNIE_IMAGE_TURBO_REPO,
    "downloadFiles": ERNIE_IMAGE_TURBO_DIFFUSERS_FILES,
    "artifactLabel": "Apache-2.0 bfloat16 Diffusers safetensors repo",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 8,
    "recommendedGuidance": 1.0,
    "recommendedMaxSequenceLength": 2048,
    "guidanceLabel": "Guidance",
    "supportsNegativePrompt": False,
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
        "steps": 8,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["text_to_image"],
    "modeRequirements": {},
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(ERNIE_IMAGE_TURBO_REPO, model_type="ErnieImagePipeline")],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The immutable public Apache-2.0 snapshot uses only package-owned Diffusers and Transformers classes and five bfloat16 safetensors weight files.",
        "The reviewed Turbo route is fixed to 1024x1024, 8 steps, guidance 1, the repository's optional prompt enhancer, and the tokenizer's 2048-token ceiling.",
        "The approximately 31.60 GB weight surface is remote-only; the missing safety checker keeps Auto and Gallery disabled pending live output review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["ernie-image-turbo:text-to-image:v1"] = {
    "modelType": "ErnieImagePipeline",
    "mode": "text_to_image",
    "profile": _ERNIE_IMAGE_TURBO_PROFILE,
    "capability": _ERNIE_IMAGE_TURBO_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _SDXL_GRAPH_BINDINGS,
}


_GLM_IMAGE_PROFILE = {
    "id": "glm-image:direct",
    "model_type": "GlmImagePipeline",
    "modes": ("text_to_image",),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "GlmImagePipeline",
    "default_repo": GLM_IMAGE_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 50,
    "live_proof": False,
    "compatible_repos": (),
}
_GLM_IMAGE_CAPABILITY = {
    "modelType": "GlmImagePipeline",
    "label": "GLM-Image",
    "displayName": "GLM-Image 16B 1024px",
    "family": "GLM-Image",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": GLM_IMAGE_REPO,
    "downloadFiles": GLM_IMAGE_DIFFUSERS_FILES,
    "artifactLabel": "MIT bfloat16 Diffusers safetensors repo",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 50,
    "recommendedGuidance": 1.5,
    "recommendedMaxSequenceLength": 2048,
    "guidanceLabel": "Guidance",
    "supportsNegativePrompt": False,
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
        "steps": 50,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["text_to_image"],
    "modeRequirements": {},
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(GLM_IMAGE_REPO, model_type="GlmImagePipeline")],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The immutable public MIT snapshot uses only package-owned Diffusers and Transformers classes and nine safetensors weight files; incorporated X-Omni tokenizer weights retain Apache-2.0 terms.",
        "The reviewed text-to-image route is fixed to 1024x1024, 50 steps, guidance 1.5, and at most 2048 prompt tokens; image-to-image remains outside this first admission.",
        "The approximately 35.77 GB weight surface is remote-only; the missing safety checker keeps Auto and Gallery disabled pending live output review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["glm-image:text-to-image:v1"] = {
    "modelType": "GlmImagePipeline",
    "mode": "text_to_image",
    "profile": _GLM_IMAGE_PROFILE,
    "capability": _GLM_IMAGE_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _SDXL_GRAPH_BINDINGS,
}


_JOYIMAGE_EDIT_PROFILE = {
    "id": "joyimage-edit:direct",
    "model_type": "JoyImageEditPipeline",
    "modes": ("text_to_image", "edit_image"),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "JoyImageEditPipeline",
    "default_repo": JOYIMAGE_EDIT_REPO,
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
_JOYIMAGE_EDIT_CAPABILITY = {
    "modelType": "JoyImageEditPipeline",
    "label": "JoyAI Image Edit",
    "displayName": "JoyAI Image Edit 16B",
    "family": "JoyAI Image",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": JOYIMAGE_EDIT_REPO,
    "downloadFiles": JOYIMAGE_EDIT_DIFFUSERS_FILES,
    "artifactLabel": "Apache-2.0 bfloat16 Diffusers safetensors repo",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 40,
    "recommendedGuidance": 4.0,
    "recommendedMaxSequenceLength": 2048,
    "guidanceLabel": "Guidance",
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
        "steps": 40,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["text_to_image", "edit_image"],
    "modeRequirements": {
        "edit_image": {
            "requiredImages": ["referenceImages"],
            "note": "Requires one source image; the package center-crops it to a reviewed 1024-base bucket.",
        }
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(JOYIMAGE_EDIT_REPO, model_type="JoyImageEditPipeline")],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The immutable public snapshot contains twelve exact bfloat16 safetensors files and executes only package-owned Diffusers and Transformers classes.",
        "MoDiff bounds the route to one 1024-base output bucket, 40 steps, guidance 4, at most 2048 prompt tokens, and one source image for edit mode.",
        "The model repository declares Apache-2.0 but omits its linked LICENSE payload; the exact upstream project Apache receipt is recorded separately.",
        "The approximately 50.32 GB weight surface is remote-only; the missing safety checker keeps Auto and Gallery disabled pending live output review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["joyimage-edit:text-to-image:v1"] = {
    "modelType": "JoyImageEditPipeline",
    "mode": "text_to_image",
    "profile": _JOYIMAGE_EDIT_PROFILE,
    "capability": _JOYIMAGE_EDIT_CAPABILITY,
    "roles": _GRAPH_ROLES,
    "edges": _GRAPH_EDGES,
    "bindings": _SDXL_GRAPH_BINDINGS,
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["joyimage-edit:edit-image:v1"] = {
    "modelType": "JoyImageEditPipeline",
    "mode": "edit_image",
    "profile": _JOYIMAGE_EDIT_PROFILE,
    "capability": _JOYIMAGE_EDIT_CAPABILITY,
    "roles": _EDIT_GRAPH_ROLES,
    "edges": _EDIT_GRAPH_EDGES,
    "bindings": _SDXL_EDIT_GRAPH_BINDINGS,
}


_JOYIMAGE_EDIT_PLUS_PROFILE = {
    "id": "joyimage-edit-plus:direct",
    "model_type": "JoyImageEditPlusPipeline",
    "modes": ("edit_image", "multi_image_reference_edit"),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "JoyImageEditPlusPipeline",
    "default_repo": JOYIMAGE_EDIT_PLUS_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": _DIRECT_OFFLOAD_MODES,
    "retry_offload_modes": (OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    "max_low_memory_side": 1024,
    "max_low_memory_steps": 30,
    "live_proof": False,
    "compatible_repos": (),
}
_JOYIMAGE_EDIT_PLUS_CAPABILITY = {
    "modelType": "JoyImageEditPlusPipeline",
    "label": "JoyAI Image Edit Plus",
    "displayName": "JoyAI Image Edit Plus 16B",
    "family": "JoyAI Image",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": JOYIMAGE_EDIT_PLUS_REPO,
    "downloadFiles": JOYIMAGE_EDIT_PLUS_DIFFUSERS_FILES,
    "artifactLabel": "Apache-2.0 bfloat16 Diffusers safetensors repo",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 30,
    "recommendedGuidance": 4.0,
    "recommendedMaxSequenceLength": 2048,
    "guidanceLabel": "Guidance",
    "supportsNegativePrompt": True,
    "supportsImageInput": True,
    "supportsMask": False,
    "supportsMultiImage": True,
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
        "steps": 30,
        "width": 1024,
        "height": 1024,
    },
    "modes": ["edit_image", "multi_image_reference_edit"],
    "modeRequirements": {
        "edit_image": {
            "requiredImages": ["referenceImages"],
            "note": "Requires one source image.",
        },
        "multi_image_reference_edit": {
            "requiredImages": ["referenceImages"],
            "note": "Requires 1 to 5 source images; each reference is independently resized to a 1024-base bucket.",
        },
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(JOYIMAGE_EDIT_PLUS_REPO, model_type="JoyImageEditPlusPipeline")],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The immutable public snapshot contains twelve exact bfloat16 safetensors files; its repository inference.py is not selected or executed.",
        "MoDiff maps the generic edit input to the package's images argument and bounds the route to 1-5 references, one 1024-base output bucket, 30 steps, guidance 4, and at most 2048 prompt tokens.",
        "The model repository declares Apache-2.0 but omits a LICENSE payload; the exact upstream project Apache receipt is recorded separately.",
        "The approximately 50.32 GB weight surface is remote-only; the missing safety checker keeps Auto and Gallery disabled pending live output review.",
    ],
}
for _spec_id, _mode in (
    ("joyimage-edit-plus:edit-image:v1", "edit_image"),
    ("joyimage-edit-plus:multi-image-reference-edit:v1", "multi_image_reference_edit"),
):
    STUDIO_EXECUTION_SPEC_DEFINITIONS[_spec_id] = {
        "modelType": "JoyImageEditPlusPipeline",
        "mode": _mode,
        "profile": _JOYIMAGE_EDIT_PLUS_PROFILE,
        "capability": _JOYIMAGE_EDIT_PLUS_CAPABILITY,
        "roles": _EDIT_GRAPH_ROLES,
        "edges": _EDIT_GRAPH_EDGES,
        "bindings": _SDXL_EDIT_GRAPH_BINDINGS,
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
    "downloadFiles": DREAMLITE_DIFFUSERS_FILES,
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
    "revisionCandidates": [require_catalog_revision(DREAMLITE_BASE_REPO, model_type="DreamLitePipeline")],
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
    "downloadFiles": DREAMLITE_DIFFUSERS_FILES,
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
    "downloadFiles": LCM_DREAMSHAPER_DIFFUSERS_FILES,
    "artifactLabel": "Diffusers safetensors repo",
    "defaultDtype": "float32",
    "defaultSize": {"width": 512, "height": 512, "aspectRatio": "1:1"},
    "recommendedSteps": 4,
    "recommendedGuidance": 8.5,
    "guidanceLabel": "Guidance",
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
        "steps": 4,
        "width": 512,
        "height": 512,
    },
    "modes": ["text_to_image", "edit_image"],
    "modeRequirements": {
        "edit_image": {
            "requiredImages": ["referenceImages"],
            "note": "Requires one source image for latent-consistency image-to-image transformation.",
        }
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(LCM_DREAMSHAPER_REPO, model_type="LatentConsistencyModelPipeline")
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The exact LCM checkpoint supports one-to-four-step text-to-image and image-to-image generation through generic image nodes.",
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
_LCM_IMG2IMG_PROFILE = {
    **_LCM_PROFILE,
    "id": "lcm-dreamshaper-v7:img2img-direct",
    "modes": ("edit_image",),
    "pipeline_class": "LatentConsistencyModelImg2ImgPipeline",
    "live_proof": False,
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["lcm-dreamshaper-v7:edit-image:v1"] = {
    "modelType": "LatentConsistencyModelPipeline",
    "mode": "edit_image",
    "profile": _LCM_IMG2IMG_PROFILE,
    "capability": _LCM_CAPABILITY,
    "roles": _EDIT_GRAPH_ROLES,
    "edges": _EDIT_GRAPH_EDGES,
    "bindings": _LCM_EDIT_GRAPH_BINDINGS,
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
    "downloadFiles": SD15_SHARED_DIFFUSERS_FILES,
    "artifactLabel": "Diffusers safetensors repo",
    "defaultDtype": "float32",
    "defaultSize": {"width": 512, "height": 512, "aspectRatio": "1:1"},
    "recommendedSteps": 30,
    "recommendedGuidance": 7.5,
    "guidanceLabel": "Guidance",
    "supportsImageInput": True,
    "supportsMask": True,
    "supportsMultiImage": False,
    "supportsControlImage": True,
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
    "modes": ["text_to_image", "edit_image", "inpaint", "control_image", "control_inpaint"],
    "modeRequirements": {
        "edit_image": {
            "requiredImages": ["referenceImages"],
            "note": "Requires one source image for PAG image-to-image transformation.",
        },
        "inpaint": {
            "requiredImages": ["referenceImages", "maskImage"],
            "note": "Requires one source image and one mask image for PAG inpainting.",
        },
        "control_image": {
            "modelRequirements": studio_model_requirements_for_pair("StableDiffusionPAGPipeline", "control_image"),
            "requiredImages": ["controlImage"],
            "note": "Requires one control image and the immutable Canny ControlNet component.",
        },
        "control_inpaint": {
            "modelRequirements": studio_model_requirements_for_pair("StableDiffusionPAGPipeline", "control_inpaint"),
            "requiredImages": ["referenceImages", "maskImage", "controlImage"],
            "note": (
                "Requires one source image, one mask, one control image, and the immutable Canny ControlNet component."
            ),
        },
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [require_catalog_revision(SD15_BASE_REPO, model_type="StableDiffusionPAGPipeline")],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "Perturbed-attention guidance text, image-to-image, and inpaint modes reuse the immutable Stable Diffusion 1.5 safetensors base.",
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
_PAG_IMG2IMG_PROFILE = {
    **_PAG_PROFILE,
    "id": "sd15-pag:img2img-direct",
    "modes": ("edit_image",),
    "pipeline_class": "StableDiffusionPAGImg2ImgPipeline",
    "live_proof": False,
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["sd15-pag:edit-image:v1"] = {
    "modelType": "StableDiffusionPAGPipeline",
    "mode": "edit_image",
    "profile": _PAG_IMG2IMG_PROFILE,
    "capability": _PAG_CAPABILITY,
    "roles": _EDIT_GRAPH_ROLES,
    "edges": _EDIT_GRAPH_EDGES,
    "bindings": _SD15_PAG_EDIT_GRAPH_BINDINGS,
}
_PAG_INPAINT_PROFILE = {
    **_PAG_PROFILE,
    "id": "sd15-pag:inpaint-direct",
    "modes": ("inpaint",),
    "pipeline_class": "StableDiffusionPAGInpaintPipeline",
    "live_proof": False,
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["sd15-pag:inpaint:v1"] = {
    "modelType": "StableDiffusionPAGPipeline",
    "mode": "inpaint",
    "profile": _PAG_INPAINT_PROFILE,
    "capability": _PAG_CAPABILITY,
    "roles": _INPAINT_GRAPH_ROLES,
    "edges": _INPAINT_GRAPH_EDGES,
    "bindings": _SD15_PAG_INPAINT_GRAPH_BINDINGS,
}
_PAG_CONTROLNET_PROFILE = {
    **_PAG_PROFILE,
    "id": "sd15-pag-controlnet-canny:direct",
    "modes": ("control_image",),
    "pipeline_class": "StableDiffusionControlNetPAGPipeline",
    "live_proof": False,
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["sd15-pag-controlnet-canny:control-image:v1"] = {
    "modelType": "StableDiffusionPAGPipeline",
    "mode": "control_image",
    "profile": _PAG_CONTROLNET_PROFILE,
    "capability": _PAG_CAPABILITY,
    "roles": _CONDITIONED_CONTROL_GRAPH_ROLES,
    "edges": _CONDITIONED_CONTROL_GRAPH_EDGES,
    "bindings": _PAG_CONDITIONED_CONTROL_GRAPH_BINDINGS,
}
_PAG_CONTROLNET_INPAINT_PROFILE = {
    **_PAG_PROFILE,
    "id": "sd15-pag-controlnet-canny:inpaint-direct",
    "modes": ("control_inpaint",),
    "pipeline_class": "StableDiffusionControlNetPAGInpaintPipeline",
    "live_proof": False,
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["sd15-pag-controlnet-canny:control-inpaint:v1"] = {
    "modelType": "StableDiffusionPAGPipeline",
    "mode": "control_inpaint",
    "profile": _PAG_CONTROLNET_INPAINT_PROFILE,
    "capability": _PAG_CAPABILITY,
    "roles": _CONDITIONED_CONTROL_INPAINT_GRAPH_ROLES,
    "edges": _CONDITIONED_CONTROL_INPAINT_GRAPH_EDGES,
    "bindings": _PAG_CONDITIONED_CONTROL_INPAINT_GRAPH_BINDINGS,
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
    "downloadFiles": MARIGOLD_DEPTH_DIFFUSERS_FILES,
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
    "revisionCandidates": [require_catalog_revision(MARIGOLD_DEPTH_LCM_REPO, model_type="MarigoldDepthPipeline")],
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

_SMOLLM2_135M_INSTRUCT_PROFILE = {
    "id": "smollm2-135m-instruct:direct",
    "model_type": "HuggingFaceTextGenerationModel",
    "modes": ("text_generation",),
    "loader_module": "modules.HuggingFaceTransformers",
    "loader_action": "LoadTextGenerationModel",
    "execution_path": "direct-huggingface-transformers-text",
    "pipeline_class": "AutoModelForCausalLM",
    "default_repo": SMOLLM2_135M_INSTRUCT_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": (OFFLOAD_MODE_NONE,),
    "retry_offload_modes": (),
    "max_low_memory_side": None,
    "max_low_memory_steps": None,
    "live_proof": False,
    "compatible_repos": (),
}
_SMOLLM2_135M_INSTRUCT_CAPABILITY = {
    "modelType": "HuggingFaceTextGenerationModel",
    "label": "SmolLM2 135M Instruct",
    "displayName": "SmolLM2 135M Instruct",
    "family": "SmolLM",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": SMOLLM2_135M_INSTRUCT_REPO,
    "artifactLabel": "Transformers safetensors repo",
    "downloadFiles": SMOLLM2_135M_INSTRUCT_TRANSFORMERS_FILES,
    "defaultDtype": "float32",
    "defaultSize": {"width": 1, "height": 1, "aspectRatio": "text"},
    "recommendedSteps": 1,
    "recommendedGuidance": 0.0,
    "guidanceLabel": "Not used",
    "supportsNegativePrompt": False,
    "supportsImageInput": False,
    "supportsAudioInput": False,
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
    "modes": ["text_generation"],
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(
            SMOLLM2_135M_INSTRUCT_REPO,
            model_type="HuggingFaceTextGenerationModel",
        )
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The generic causal-LM boundary clamps input and output tokens and returns a versioned JSON receipt plus plain text.",
        "Expert bnb_4bit is bounded to this exact revision with bfloat16 compute on an NVIDIA CUDA runtime that delivers bitsandbytes 0.50.0.",
        "Auto and Gallery remain disabled until app-installed weights receive live output review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["smollm2-135m-instruct:text-generation:v1"] = {
    "modelType": "HuggingFaceTextGenerationModel",
    "mode": "text_generation",
    "profile": _SMOLLM2_135M_INSTRUCT_PROFILE,
    "capability": _SMOLLM2_135M_INSTRUCT_CAPABILITY,
    "roles": _TRANSFORMERS_TEXT_GRAPH_ROLES,
    "edges": _TRANSFORMERS_TEXT_GRAPH_EDGES,
    "bindings": _TRANSFORMERS_TEXT_GRAPH_BINDINGS,
}

_SMOLVLM_256M_INSTRUCT_PROFILE = {
    "id": "smolvlm-256m-instruct:direct",
    "model_type": "HuggingFaceImageTextToTextModel",
    "modes": ("image_to_text",),
    "loader_module": "modules.HuggingFaceTransformers",
    "loader_action": "LoadImageTextToTextModel",
    "execution_path": "direct-huggingface-transformers-image-text",
    "pipeline_class": "AutoModelForImageTextToText",
    "default_repo": SMOLVLM_256M_INSTRUCT_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": (OFFLOAD_MODE_NONE,),
    "retry_offload_modes": (),
    "max_low_memory_side": 4096,
    "max_low_memory_steps": None,
    "live_proof": False,
    "compatible_repos": (),
}
_SMOLVLM_256M_INSTRUCT_CAPABILITY = {
    "modelType": "HuggingFaceImageTextToTextModel",
    "label": "SmolVLM 256M Instruct",
    "displayName": "SmolVLM 256M Instruct",
    "family": "SmolVLM",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": SMOLVLM_256M_INSTRUCT_REPO,
    "artifactLabel": "Transformers safetensors repo",
    "downloadFiles": SMOLVLM_256M_INSTRUCT_TRANSFORMERS_FILES,
    "defaultDtype": "float32",
    "defaultSize": {"width": 512, "height": 512, "aspectRatio": "1:1"},
    "recommendedSteps": 1,
    "recommendedGuidance": 0.0,
    "guidanceLabel": "Not used",
    "supportsNegativePrompt": False,
    "supportsImageInput": True,
    "supportsAudioInput": False,
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
        "width": 512,
        "height": 512,
    },
    "modes": ["image_to_text"],
    "modeRequirements": {
        "image_to_text": {
            "requiredImages": ["referenceImages"],
            "note": "Requires exactly one bounded local image and returns bounded text with a versioned receipt.",
        }
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(
            SMOLVLM_256M_INSTRUCT_REPO,
            model_type="HuggingFaceImageTextToTextModel",
        )
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The generic image-to-text boundary clamps media count, image geometry, input tokens, and generated tokens.",
        "Auto and Gallery remain disabled until app-installed weights receive live output review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["smolvlm-256m-instruct:image-to-text:v1"] = {
    "modelType": "HuggingFaceImageTextToTextModel",
    "mode": "image_to_text",
    "profile": _SMOLVLM_256M_INSTRUCT_PROFILE,
    "capability": _SMOLVLM_256M_INSTRUCT_CAPABILITY,
    "roles": _TRANSFORMERS_IMAGE_TEXT_GRAPH_ROLES,
    "edges": _TRANSFORMERS_IMAGE_TEXT_GRAPH_EDGES,
    "bindings": _TRANSFORMERS_IMAGE_TEXT_GRAPH_BINDINGS,
}

_JANUS_PRO_1B_PROFILE = {
    "id": "janus-pro-1b:direct",
    "model_type": "HuggingFaceAnyToAnyModel",
    "modes": ("text_generation", "image_to_text", "text_to_image"),
    "loader_module": "modules.HuggingFaceTransformers",
    "loader_action": "LoadAnyToAnyModel",
    "execution_path": "direct-huggingface-transformers-any-to-any",
    "pipeline_class": "JanusForConditionalGeneration",
    "default_repo": JANUS_PRO_1B_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": (OFFLOAD_MODE_NONE,),
    "retry_offload_modes": (),
    "max_low_memory_side": 384,
    "max_low_memory_steps": None,
    "live_proof": False,
    "compatible_repos": (),
}
_JANUS_PRO_1B_CAPABILITY = {
    "modelType": "HuggingFaceAnyToAnyModel",
    "label": "Janus Pro 1B",
    "displayName": "Janus-Pro-1B",
    "family": "Janus",
    "surfaceCategory": "Utility",
    "runtimeKind": "transformers",
    "isDiffusersBacked": False,
    "catalogVisibility": "workflowOnly",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": JANUS_PRO_1B_REPO,
    "artifactLabel": "Transformers safetensors repo (DeepSeek Model License)",
    "downloadFiles": JANUS_PRO_1B_TRANSFORMERS_FILES,
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 384, "height": 384, "aspectRatio": "1:1"},
    "recommendedSteps": 1,
    "recommendedGuidance": 0.0,
    "guidanceLabel": "Not used",
    "supportsNegativePrompt": False,
    "supportsImageInput": True,
    "supportsAudioInput": False,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": False,
    "outputKind": "image",
    "modeOutputKinds": {
        "text_generation": "json",
        "image_to_text": "json",
        "text_to_image": "image",
    },
    "offloadSupport": {
        "default": OFFLOAD_MODE_NONE,
        "lowVram": OFFLOAD_MODE_NONE,
        "emergency": OFFLOAD_MODE_NONE,
        "modes": [OFFLOAD_MODE_NONE],
    },
    "lowVram": {
        "dtype": "bfloat16",
        "autoOffload": False,
        "offloadMode": OFFLOAD_MODE_NONE,
        "steps": 1,
        "width": 384,
        "height": 384,
    },
    "modes": ["text_generation", "image_to_text", "text_to_image"],
    "modeRequirements": {
        "text_generation": {
            "note": "Accepts one bounded prompt and returns bounded text with a versioned receipt.",
        },
        "image_to_text": {
            "requiredImages": ["referenceImages"],
            "note": "Requires exactly one bounded local image and returns bounded text with a versioned receipt.",
        },
        "text_to_image": {
            "note": "Accepts one bounded prompt and returns the native reviewed 384x384 image output.",
        },
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(
            JANUS_PRO_1B_REPO,
            model_type="HuggingFaceAnyToAnyModel",
        )
    ],
    "license": "DeepSeek Model License Agreement v1.0",
    "licenseCompliance": {
        "state": "product_and_user_review_required",
        "codeLicense": "MIT",
        "weightsLicense": "DeepSeek Model License Agreement v1.0",
        "noticePath": "licenses/DeepSeek-Model-License-1.0.txt",
        "useRestrictionsPresent": True,
        "distributionAndHostedUseCarryDuties": True,
        "sourceExecutable": True,
        "liveExecutionQualified": False,
    },
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The native Janus adapter is finite, local-files-only, safetensors-only, and emits exact text or image schemas.",
        "The app must preflight free space before installing the immutable 4.161 GB selection; no existing model may be deleted.",
        "Auto, Gallery, and live execution claims remain disabled pending real-weight output and product/user license review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["janus-pro-1b:text-generation:v1"] = {
    "modelType": "HuggingFaceAnyToAnyModel",
    "mode": "text_generation",
    "profile": _JANUS_PRO_1B_PROFILE,
    "capability": _JANUS_PRO_1B_CAPABILITY,
    "roles": _TRANSFORMERS_ANY_TO_ANY_TEXT_GRAPH_ROLES,
    "edges": _TRANSFORMERS_ANY_TO_ANY_TEXT_GRAPH_EDGES,
    "bindings": _TRANSFORMERS_ANY_TO_ANY_TEXT_GRAPH_BINDINGS,
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["janus-pro-1b:image-to-text:v1"] = {
    "modelType": "HuggingFaceAnyToAnyModel",
    "mode": "image_to_text",
    "profile": _JANUS_PRO_1B_PROFILE,
    "capability": _JANUS_PRO_1B_CAPABILITY,
    "roles": _TRANSFORMERS_ANY_TO_ANY_IMAGE_TEXT_GRAPH_ROLES,
    "edges": _TRANSFORMERS_ANY_TO_ANY_IMAGE_TEXT_GRAPH_EDGES,
    "bindings": _TRANSFORMERS_ANY_TO_ANY_IMAGE_TEXT_GRAPH_BINDINGS,
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["janus-pro-1b:text-to-image:v1"] = {
    "modelType": "HuggingFaceAnyToAnyModel",
    "mode": "text_to_image",
    "profile": _JANUS_PRO_1B_PROFILE,
    "capability": _JANUS_PRO_1B_CAPABILITY,
    "roles": _TRANSFORMERS_ANY_TO_ANY_IMAGE_GRAPH_ROLES,
    "edges": _TRANSFORMERS_ANY_TO_ANY_IMAGE_GRAPH_EDGES,
    "bindings": _TRANSFORMERS_ANY_TO_ANY_IMAGE_GRAPH_BINDINGS,
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
    # Direct-node qualification predates the Cluster surface. Keep the
    # immutable Cluster publication fail-closed until its own visible-frontend
    # transcription/translation evidence receives manual review.
    "live_proof": False,
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
    "downloadFiles": WHISPER_TINY_TRANSFORMERS_FILES,
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
_WAV2VEC2_BASE_960H_PROFILE = {
    "id": "wav2vec2-base-960h:ctc-direct",
    "model_type": "HuggingFaceCTCSpeechRecognitionModel",
    "modes": ("speech_to_text",),
    "loader_module": "modules.HuggingFaceSpeech",
    "loader_action": "LoadCTCSpeechRecognitionModel",
    "execution_path": "direct-huggingface-speech-ctc",
    "pipeline_class": "AutoModelForCTC",
    "default_repo": WAV2VEC2_BASE_960H_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": (OFFLOAD_MODE_NONE,),
    "retry_offload_modes": (),
    "max_low_memory_side": None,
    "max_low_memory_steps": None,
    "live_proof": False,
    "compatible_repos": (),
}
_WAV2VEC2_BASE_960H_CAPABILITY = {
    "modelType": "HuggingFaceCTCSpeechRecognitionModel",
    "label": "Wav2Vec2 Base 960h",
    "displayName": "Wav2Vec2 Base 960h",
    "family": "Wav2Vec2 CTC",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": WAV2VEC2_BASE_960H_REPO,
    "artifactLabel": "Transformers safetensors repo",
    "downloadFiles": WAV2VEC2_BASE_960H_TRANSFORMERS_FILES,
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
    "modes": ["speech_to_text"],
    "modeRequirements": {
        "speech_to_text": {
            "requiredAudio": ["sourceAudio"],
            "note": "Requires one bounded local audio source and returns an English CTC transcript.",
        },
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(
            WAV2VEC2_BASE_960H_REPO,
            model_type="HuggingFaceCTCSpeechRecognitionModel",
        )
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "notes": [
        "The CTC route exposes transcription only; Whisper task/language generation controls are absent.",
        "Auto, Gallery, and live execution claims remain disabled pending visible-frontend qualification review.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["wav2vec2-base-960h:speech-to-text:v1"] = {
    "modelType": "HuggingFaceCTCSpeechRecognitionModel",
    "mode": "speech_to_text",
    "profile": _WAV2VEC2_BASE_960H_PROFILE,
    "capability": _WAV2VEC2_BASE_960H_CAPABILITY,
    "roles": _CTC_SPEECH_GRAPH_ROLES,
    "edges": _CTC_SPEECH_GRAPH_EDGES,
    "bindings": _CTC_SPEECH_GRAPH_BINDINGS,
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["flux-redux:multi-image-reference-edit:v1"] = {
    "modelType": "FluxReduxPipeline",
    "mode": "multi_image_reference_edit",
    "profile": _flux_redux_profile(),
    "roles": _EDIT_GRAPH_ROLES,
    "edges": _EDIT_GRAPH_EDGES,
    "bindings": _EDIT_GRAPH_BINDINGS,
}
for _flux_combined_spec_id, _flux_combined_definition in _FLUX_COMBINED_CONTROL_DEFINITIONS.items():
    _flux_control_family = _flux_combined_spec_id.split(":", 1)[0]
    _flux_control_base = STUDIO_EXECUTION_SPEC_DEFINITIONS[f"{_flux_control_family}:control-image:v1"]
    _flux_combined_definition["capability"] = _flux_control_base["capability"]
    _flux_combined_definition["autoRequirements"] = _flux_control_base["autoRequirements"]
    STUDIO_EXECUTION_SPEC_DEFINITIONS[_flux_combined_spec_id] = _flux_combined_definition


def _direct_image_promotion_profile(
    *,
    profile_id: str,
    model_type: str,
    modes: tuple[str, ...],
    pipeline_class: str,
    repository: str,
    quantizable_components: tuple[str, ...],
    default_quantized_components: tuple[str, ...],
    supported_offload_modes: tuple[str, ...],
    retry_offload_modes: tuple[str, ...],
    max_low_memory_side: int,
    max_low_memory_steps: int,
    compatible_repos: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "id": profile_id,
        "model_type": model_type,
        "modes": modes,
        "loader_module": "modules.DiffusersImage",
        "loader_action": "LoadPipeline",
        "execution_path": "direct-diffusers-image",
        "pipeline_class": pipeline_class,
        "default_repo": repository,
        "fallback_repo": None,
        "quantizable_components": quantizable_components,
        "default_quantized_components": default_quantized_components,
        "supported_offload_modes": supported_offload_modes,
        "retry_offload_modes": retry_offload_modes,
        "max_low_memory_side": max_low_memory_side,
        "max_low_memory_steps": max_low_memory_steps,
        "live_proof": False,
        "compatible_repos": compatible_repos,
    }


def _direct_image_promotion_capability(
    *,
    model_type: str,
    label: str,
    display_name: str,
    family: str,
    repository: str,
    download_files: list[str],
    modes: tuple[str, ...],
    mode_requirements: dict[str, dict[str, Any]],
    supports_negative_prompt: bool,
    supports_mask: bool,
    supports_multi_image: bool,
    recommended_steps: int,
    recommended_guidance: float,
    low_vram_side: int,
    low_vram_steps: int,
    supported_offload_modes: tuple[str, ...],
    low_vram_offload_mode: str,
    notes: list[str],
    alternate_artifact: str | None = None,
) -> dict[str, Any]:
    return {
        "modelType": model_type,
        "label": label,
        "displayName": display_name,
        "family": family,
        "supportTier": "supported",
        "qualificationStatus": "graph-qualified-execution-pending",
        "qualifiedModes": [],
        "defaultRepo": repository,
        "downloadFiles": download_files,
        **({"alternateArtifact": alternate_artifact} if alternate_artifact else {}),
        "artifactLabel": "Reused immutable safetensors Diffusers repository",
        "defaultDtype": "bfloat16",
        "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
        "recommendedSteps": recommended_steps,
        "recommendedGuidance": recommended_guidance,
        "guidanceLabel": "Guidance",
        "supportsNegativePrompt": supports_negative_prompt,
        "supportsImageInput": True,
        "supportsMask": supports_mask,
        "supportsMultiImage": supports_multi_image,
        "supportsControlImage": False,
        "supportsLayers": False,
        "supportsLora": True,
        "outputKind": "image",
        "offloadSupport": {
            "default": OFFLOAD_MODE_MODEL_CPU,
            "lowVram": low_vram_offload_mode,
            "emergency": OFFLOAD_MODE_GROUP_DISK,
            "modes": list(supported_offload_modes),
        },
        "lowVram": {
            "dtype": "bfloat16",
            "autoOffload": True,
            "offloadMode": low_vram_offload_mode,
            "steps": low_vram_steps,
            "width": low_vram_side,
            "height": low_vram_side,
        },
        "modes": list(modes),
        "modeRequirements": mode_requirements,
        "executionStatus": "expert_only",
        "revisionCandidates": [require_catalog_revision(repository)],
        "autoEligible": False,
        # Source-executable pairs may be contracted for hidden authoring while
        # public Gallery publication remains a separate qualification gate.
        "templateEligible": True,
        "galleryEligible": False,
        "liveProof": False,
        "notes": notes,
    }


_QWEN_IMAGE_EDIT_DIRECT_PROFILE = _direct_image_promotion_profile(
    profile_id="qwen-image-edit:direct",
    model_type="QwenImageEditPipeline",
    modes=("edit_image",),
    pipeline_class="QwenImageEditPipeline",
    repository=QWEN_IMAGE_EDIT_REPO,
    quantizable_components=("transformer", "text_encoder"),
    default_quantized_components=(),
    supported_offload_modes=_DIRECT_OFFLOAD_MODES,
    retry_offload_modes=(
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_SEQUENTIAL_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ),
    max_low_memory_side=768,
    max_low_memory_steps=24,
)
_QWEN_IMAGE_EDIT_PLUS_DIRECT_PROFILE = _direct_image_promotion_profile(
    profile_id="qwen-image-edit-plus:direct",
    model_type="QwenImageEditPlusPipeline",
    modes=("edit_image", "multi_image_reference_edit"),
    pipeline_class="QwenImageEditPlusPipeline",
    repository=QWEN_IMAGE_EDIT_PLUS_REPO,
    quantizable_components=("transformer", "text_encoder"),
    default_quantized_components=(),
    supported_offload_modes=_DIRECT_OFFLOAD_MODES,
    retry_offload_modes=(
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_SEQUENTIAL_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ),
    max_low_memory_side=768,
    max_low_memory_steps=24,
)
_Z_IMAGE_INPAINT_DIRECT_PROFILE = _direct_image_promotion_profile(
    profile_id="z-image-inpaint:direct",
    model_type="ZImageInpaintPipeline",
    modes=("inpaint", "outpaint"),
    pipeline_class="ZImageInpaintPipeline",
    repository=Z_IMAGE_REPO,
    quantizable_components=(),
    default_quantized_components=(),
    supported_offload_modes=(
        OFFLOAD_MODE_NONE,
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ),
    retry_offload_modes=(OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK),
    max_low_memory_side=1024,
    max_low_memory_steps=8,
)
_FLUX_KONTEXT_INPAINT_DIRECT_PROFILE = _direct_image_promotion_profile(
    profile_id="flux-kontext-inpaint:direct",
    model_type="FluxKontextInpaintPipeline",
    modes=("inpaint", "outpaint"),
    pipeline_class="FluxKontextInpaintPipeline",
    repository=FLUX_KONTEXT_REPO,
    quantizable_components=("transformer", "text_encoder_2"),
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
    compatible_repos=(FLUX_KONTEXT_NVFP4_REPO,),
)
_FLUX2_KLEIN_INPAINT_DIRECT_PROFILE = _direct_image_promotion_profile(
    profile_id="flux2-klein-inpaint:direct",
    model_type="Flux2KleinInpaintPipeline",
    modes=("inpaint", "outpaint"),
    pipeline_class="Flux2KleinInpaintPipeline",
    repository=FLUX2_KLEIN_REPO,
    # The exact standard inpaint constructor owns one text_encoder component,
    # unlike the older base route's historical text_encoder_2 declaration.
    quantizable_components=("transformer", "text_encoder"),
    default_quantized_components=("transformer",),
    supported_offload_modes=(
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_SEQUENTIAL_CPU,
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ),
    retry_offload_modes=(OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK),
    max_low_memory_side=768,
    max_low_memory_steps=4,
)
_CHROMA1_HD_IMG2IMG_DIRECT_PROFILE = _direct_image_promotion_profile(
    profile_id="chroma1-hd-img2img:direct",
    model_type="ChromaImg2ImgPipeline",
    modes=("edit_image",),
    pipeline_class="ChromaImg2ImgPipeline",
    repository=CHROMA1_HD_REPO,
    quantizable_components=(),
    default_quantized_components=(),
    supported_offload_modes=_DIRECT_OFFLOAD_MODES,
    retry_offload_modes=(OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    max_low_memory_side=1024,
    max_low_memory_steps=40,
)
_CHROMA1_HD_INPAINT_DIRECT_PROFILE = _direct_image_promotion_profile(
    profile_id="chroma1-hd-inpaint:direct",
    model_type="ChromaInpaintPipeline",
    modes=("inpaint", "outpaint"),
    pipeline_class="ChromaInpaintPipeline",
    repository=CHROMA1_HD_REPO,
    quantizable_components=(),
    default_quantized_components=(),
    supported_offload_modes=_DIRECT_OFFLOAD_MODES,
    retry_offload_modes=(OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU),
    max_low_memory_side=1024,
    max_low_memory_steps=40,
)
_LTX2_STANDARD_DIRECT_PROFILE = _planning_video_profile(
    "ltx2-standard:direct",
    "LTX2Pipeline",
    ("text_to_video",),
    "LTX2Pipeline",
    LTX2_REPO,
)

_QWEN_IMAGE_EDIT_DIRECT_CAPABILITY = _direct_image_promotion_capability(
    model_type="QwenImageEditPipeline",
    label="Qwen Image Edit (Standard Diffusers)",
    display_name="Qwen-Image-Edit",
    family="Qwen Image",
    repository=QWEN_IMAGE_EDIT_REPO,
    download_files=QWEN_IMAGE_EDIT_DIFFUSERS_FILES,
    modes=("edit_image",),
    mode_requirements={
        "edit_image": {
            "requiredImages": ["referenceImages"],
            "note": "Requires one source image for the exact standard Qwen edit pipeline.",
        }
    },
    supports_negative_prompt=True,
    supports_mask=False,
    supports_multi_image=False,
    recommended_steps=40,
    recommended_guidance=4.0,
    low_vram_side=768,
    low_vram_steps=24,
    supported_offload_modes=_DIRECT_OFFLOAD_MODES,
    low_vram_offload_mode=OFFLOAD_MODE_SEQUENTIAL_CPU,
    notes=[
        "This additive standard-pipeline pair does not replace the approved Qwen Modular edit workflow.",
        "Auto, Gallery publication, and live qualification remain disabled pending reviewed generated assets.",
    ],
)
_QWEN_IMAGE_EDIT_PLUS_DIRECT_CAPABILITY = _direct_image_promotion_capability(
    model_type="QwenImageEditPlusPipeline",
    label="Qwen Image Edit Plus (Standard Diffusers)",
    display_name="Qwen-Image-Edit-2511",
    family="Qwen Image",
    repository=QWEN_IMAGE_EDIT_PLUS_REPO,
    download_files=QWEN_IMAGE_EDIT_2511_DIFFUSERS_FILES,
    modes=("edit_image", "multi_image_reference_edit"),
    mode_requirements={
        "edit_image": {
            "requiredImages": ["referenceImages"],
            "note": "Requires one source image for the exact standard Qwen Edit Plus pipeline.",
        },
        "multi_image_reference_edit": {
            "requiredImages": ["referenceImages"],
            "note": "Requires two or more source images and accepts at most eight reviewed references.",
        },
    },
    supports_negative_prompt=True,
    supports_mask=False,
    supports_multi_image=True,
    recommended_steps=40,
    recommended_guidance=4.0,
    low_vram_side=768,
    low_vram_steps=24,
    supported_offload_modes=_DIRECT_OFFLOAD_MODES,
    low_vram_offload_mode=OFFLOAD_MODE_SEQUENTIAL_CPU,
    notes=[
        "These additive standard-pipeline pairs do not replace the approved Qwen Edit Plus Modular workflows.",
        "Auto, Gallery publication, and live qualification remain disabled pending reviewed generated assets.",
    ],
)
_Z_IMAGE_INPAINT_DIRECT_CAPABILITY = _direct_image_promotion_capability(
    model_type="ZImageInpaintPipeline",
    label="Z-Image Inpaint (Standard Diffusers)",
    display_name="Z-Image-Turbo Inpaint",
    family="Z-Image",
    repository=Z_IMAGE_REPO,
    download_files=Z_IMAGE_DIFFUSERS_FILES,
    modes=("inpaint", "outpaint"),
    mode_requirements={
        "inpaint": {
            "requiredImages": ["referenceImages", "maskImage"],
            "note": "Requires one source image and one mask image.",
        },
        "outpaint": {
            "requiredImages": ["referenceImages"],
            "note": "Requires one source image; the generic canvas node derives the expansion mask.",
        },
    },
    supports_negative_prompt=True,
    supports_mask=True,
    supports_multi_image=False,
    recommended_steps=8,
    recommended_guidance=1.0,
    low_vram_side=1024,
    low_vram_steps=8,
    supported_offload_modes=(
        OFFLOAD_MODE_NONE,
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ),
    low_vram_offload_mode=OFFLOAD_MODE_MODEL_CPU,
    notes=[
        "This additive exact inpaint class leaves the existing Z-Image text and edit routes unchanged.",
        "Auto, Gallery publication, and live qualification remain disabled pending reviewed generated assets.",
    ],
)
_FLUX_KONTEXT_INPAINT_DIRECT_CAPABILITY = _direct_image_promotion_capability(
    model_type="FluxKontextInpaintPipeline",
    label="FLUX.1 Kontext Inpaint (Standard Diffusers)",
    display_name="FLUX.1-Kontext-dev Inpaint",
    family="FLUX Image",
    repository=FLUX_KONTEXT_REPO,
    download_files=FLUX_KONTEXT_DIFFUSERS_FILES,
    modes=("inpaint", "outpaint"),
    mode_requirements={
        "inpaint": {
            "requiredImages": ["referenceImages", "maskImage"],
            "note": "Requires one source image and one mask image.",
        },
        "outpaint": {
            "requiredImages": ["referenceImages"],
            "note": "Requires one source image; the generic canvas node derives the expansion mask.",
        },
    },
    supports_negative_prompt=True,
    supports_mask=True,
    supports_multi_image=False,
    recommended_steps=24,
    recommended_guidance=3.5,
    low_vram_side=768,
    low_vram_steps=24,
    supported_offload_modes=(
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_SEQUENTIAL_CPU,
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ),
    low_vram_offload_mode=OFFLOAD_MODE_SEQUENTIAL_CPU,
    alternate_artifact=FLUX_KONTEXT_NVFP4_REPO,
    notes=[
        "This additive exact inpaint class leaves the existing FLUX Kontext edit routes unchanged.",
        "Public publication, Auto, Gallery, live output, and rights review remain separate pending gates.",
    ],
)
_FLUX2_KLEIN_INPAINT_DIRECT_CAPABILITY = _direct_image_promotion_capability(
    model_type="Flux2KleinInpaintPipeline",
    label="FLUX.2 Klein Inpaint (Standard Diffusers)",
    display_name="FLUX.2-klein-4B Inpaint",
    family="FLUX Image",
    repository=FLUX2_KLEIN_REPO,
    download_files=FLUX2_KLEIN_DIFFUSERS_FILES,
    modes=("inpaint", "outpaint"),
    mode_requirements={
        "inpaint": {
            "requiredImages": ["referenceImages", "maskImage"],
            "note": "Requires one source image and one mask image.",
        },
        "outpaint": {
            "requiredImages": ["referenceImages"],
            "note": "Requires one source image; the generic canvas node derives the expansion mask.",
        },
    },
    supports_negative_prompt=False,
    supports_mask=True,
    supports_multi_image=False,
    recommended_steps=4,
    recommended_guidance=1.0,
    low_vram_side=768,
    low_vram_steps=4,
    supported_offload_modes=(
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_SEQUENTIAL_CPU,
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
    ),
    low_vram_offload_mode=OFFLOAD_MODE_MODEL_CPU,
    notes=[
        "This additive exact inpaint class leaves the existing FLUX.2 Klein generation and edit routes unchanged.",
        "Auto, Gallery publication, and live qualification remain disabled pending reviewed generated assets.",
    ],
)
_CHROMA1_HD_IMG2IMG_DIRECT_CAPABILITY = {
    **_direct_image_promotion_capability(
        model_type="ChromaImg2ImgPipeline",
        label="Chroma Image-to-Image (Standard Diffusers)",
        display_name="Chroma1-HD Image-to-Image",
        family="Chroma",
        repository=CHROMA1_HD_REPO,
        download_files=CHROMA1_HD_DIFFUSERS_FILES,
        modes=("edit_image",),
        mode_requirements={
            "edit_image": {
                "requiredImages": ["referenceImages"],
                "note": "Requires exactly one source image for the exact standard Chroma image-to-image pipeline.",
            }
        },
        supports_negative_prompt=True,
        supports_mask=False,
        supports_multi_image=False,
        recommended_steps=35,
        recommended_guidance=5.0,
        low_vram_side=1024,
        low_vram_steps=35,
        supported_offload_modes=_DIRECT_OFFLOAD_MODES,
        low_vram_offload_mode=OFFLOAD_MODE_SEQUENTIAL_CPU,
        notes=[
            "This additive exact image-to-image class leaves the existing Chroma text-to-image route unchanged.",
            "Auto, Gallery publication, and live qualification remain disabled pending reviewed generated assets.",
        ],
    ),
    "artifactLabel": "Reused Apache-2.0 bfloat16 safetensors Diffusers repository",
    "recommendedMaxSequenceLength": 512,
    "supportsLora": False,
}
_CHROMA1_HD_INPAINT_DIRECT_CAPABILITY = {
    **_direct_image_promotion_capability(
        model_type="ChromaInpaintPipeline",
        label="Chroma Inpaint (Standard Diffusers)",
        display_name="Chroma1-HD Inpaint",
        family="Chroma",
        repository=CHROMA1_HD_REPO,
        download_files=CHROMA1_HD_DIFFUSERS_FILES,
        modes=("inpaint", "outpaint"),
        mode_requirements={
            "inpaint": {
                "requiredImages": ["referenceImages", "maskImage"],
                "note": "Requires exactly one source image and one mask image.",
            },
            "outpaint": {
                "requiredImages": ["referenceImages"],
                "note": "Requires one source image; the generic canvas node derives the expansion mask.",
            },
        },
        supports_negative_prompt=True,
        supports_mask=True,
        supports_multi_image=False,
        recommended_steps=28,
        recommended_guidance=7.0,
        low_vram_side=1024,
        low_vram_steps=28,
        supported_offload_modes=_DIRECT_OFFLOAD_MODES,
        low_vram_offload_mode=OFFLOAD_MODE_SEQUENTIAL_CPU,
        notes=[
            "This additive exact inpaint class leaves the existing Chroma text-to-image route unchanged.",
            "The inert upstream true_cfg_scale field is rejected; the generic guidance_scale contract is authoritative.",
            "Auto, Gallery publication, and live qualification remain disabled pending reviewed generated assets.",
        ],
    ),
    "artifactLabel": "Reused Apache-2.0 bfloat16 safetensors Diffusers repository",
    "recommendedMaxSequenceLength": 512,
    "supportsLora": False,
}
_LTX2_STANDARD_DIRECT_CAPABILITY = {
    **_planning_video_capability(
        "LTX2Pipeline",
        "LTX-2 Standard Text-to-Video + Audio",
        "LTX Video",
        LTX2_REPO,
        ("text_to_video",),
        {
            "text_to_video": {
                "note": "Text-only generation returns both the synchronized video stream and its audio stream."
            }
        },
    ),
    "artifactLabel": "Reused immutable LTX-2 safetensors Diffusers repository",
    "recommendedSteps": 40,
    "recommendedGuidance": 4.0,
    "recommendedFrames": 121,
    "outputMedia": ["video", "audio"],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "liveProof": False,
    "notes": [
        "This additive exact standard class leaves every existing LTX-2 condition workflow unchanged.",
        "The immutable artifact is reused without a new download selection; its license and large-weight review remain explicit gates.",
        "Auto, Gallery publication, and live qualification remain disabled pending reviewed generated assets.",
    ],
}

_CHROMA_IMG2IMG_DIRECT_BINDINGS = tuple(
    item for item in _SDXL_EDIT_GRAPH_BINDINGS if item[:2] != ("diffusersImageEdit", "reference_strength")
)

_STANDARD_DIRECT_IMAGE_PROMOTION_DEFINITIONS = {
    "qwen-image-edit-direct:edit-image:v1": {
        "modelType": "QwenImageEditPipeline",
        "mode": "edit_image",
        "profile": _QWEN_IMAGE_EDIT_DIRECT_PROFILE,
        "capability": _QWEN_IMAGE_EDIT_DIRECT_CAPABILITY,
        "roles": _EDIT_GRAPH_ROLES,
        "edges": _EDIT_GRAPH_EDGES,
        "bindings": _QWEN_DIRECT_EDIT_GRAPH_BINDINGS,
    },
    "qwen-image-edit-plus-direct:edit-image:v1": {
        "modelType": "QwenImageEditPlusPipeline",
        "mode": "edit_image",
        "profile": _QWEN_IMAGE_EDIT_PLUS_DIRECT_PROFILE,
        "capability": _QWEN_IMAGE_EDIT_PLUS_DIRECT_CAPABILITY,
        "roles": _EDIT_GRAPH_ROLES,
        "edges": _EDIT_GRAPH_EDGES,
        "bindings": _QWEN_DIRECT_EDIT_GRAPH_BINDINGS,
    },
    "qwen-image-edit-plus-direct:multi-image-reference-edit:v1": {
        "modelType": "QwenImageEditPlusPipeline",
        "mode": "multi_image_reference_edit",
        "profile": _QWEN_IMAGE_EDIT_PLUS_DIRECT_PROFILE,
        "capability": _QWEN_IMAGE_EDIT_PLUS_DIRECT_CAPABILITY,
        "roles": _EDIT_GRAPH_ROLES,
        "edges": _EDIT_GRAPH_EDGES,
        "bindings": _QWEN_DIRECT_EDIT_GRAPH_BINDINGS,
    },
}

for (
    _inpaint_prefix,
    _inpaint_model_type,
    _inpaint_profile,
    _inpaint_capability,
    _inpaint_unsupported_params,
) in (
    (
        "z-image-inpaint-direct",
        "ZImageInpaintPipeline",
        _Z_IMAGE_INPAINT_DIRECT_PROFILE,
        _Z_IMAGE_INPAINT_DIRECT_CAPABILITY,
        frozenset({"reference_strength"}),
    ),
    (
        "flux-kontext-inpaint-direct",
        "FluxKontextInpaintPipeline",
        _FLUX_KONTEXT_INPAINT_DIRECT_PROFILE,
        _FLUX_KONTEXT_INPAINT_DIRECT_CAPABILITY,
        frozenset({"reference_strength"}),
    ),
    (
        "flux2-klein-inpaint-direct",
        "Flux2KleinInpaintPipeline",
        _FLUX2_KLEIN_INPAINT_DIRECT_PROFILE,
        _FLUX2_KLEIN_INPAINT_DIRECT_CAPABILITY,
        frozenset({"negative_prompt", "reference_strength"}),
    ),
):
    for _inpaint_mode in ("inpaint", "outpaint"):
        _STANDARD_DIRECT_IMAGE_PROMOTION_DEFINITIONS[f"{_inpaint_prefix}:{_inpaint_mode}:v1"] = {
            "modelType": _inpaint_model_type,
            "mode": _inpaint_mode,
            "profile": _inpaint_profile,
            "capability": _inpaint_capability,
            "roles": _INPAINT_GRAPH_ROLES if _inpaint_mode == "inpaint" else _OUTPAINT_GRAPH_ROLES,
            "edges": _INPAINT_GRAPH_EDGES if _inpaint_mode == "inpaint" else _OUTPAINT_GRAPH_EDGES,
            "bindings": _direct_inpaint_bindings(
                outpaint=_inpaint_mode == "outpaint",
                unsupported_params=_inpaint_unsupported_params,
            ),
        }

_STANDARD_DIRECT_IMAGE_PROMOTION_DEFINITIONS["chroma1-hd-img2img:edit-image:v1"] = {
    "modelType": "ChromaImg2ImgPipeline",
    "mode": "edit_image",
    "profile": _CHROMA1_HD_IMG2IMG_DIRECT_PROFILE,
    "capability": _CHROMA1_HD_IMG2IMG_DIRECT_CAPABILITY,
    "roles": _EDIT_GRAPH_ROLES,
    "edges": _EDIT_GRAPH_EDGES,
    "bindings": _CHROMA_IMG2IMG_DIRECT_BINDINGS,
}
for _chroma_inpaint_mode in ("inpaint", "outpaint"):
    _STANDARD_DIRECT_IMAGE_PROMOTION_DEFINITIONS[f"chroma1-hd-inpaint:{_chroma_inpaint_mode}:v1"] = {
        "modelType": "ChromaInpaintPipeline",
        "mode": _chroma_inpaint_mode,
        "profile": _CHROMA1_HD_INPAINT_DIRECT_PROFILE,
        "capability": _CHROMA1_HD_INPAINT_DIRECT_CAPABILITY,
        "roles": (_INPAINT_GRAPH_ROLES if _chroma_inpaint_mode == "inpaint" else _OUTPAINT_GRAPH_ROLES),
        "edges": (_INPAINT_GRAPH_EDGES if _chroma_inpaint_mode == "inpaint" else _OUTPAINT_GRAPH_EDGES),
        "bindings": _direct_inpaint_bindings(
            outpaint=_chroma_inpaint_mode == "outpaint",
            unsupported_params=frozenset({"reference_strength"}),
        ),
    }

STUDIO_EXECUTION_SPEC_DEFINITIONS.update(_STANDARD_DIRECT_IMAGE_PROMOTION_DEFINITIONS)
STUDIO_EXECUTION_SPEC_DEFINITIONS["ltx2-standard:text-to-video:v1"] = {
    "modelType": "LTX2Pipeline",
    "mode": "text_to_video",
    "profile": _LTX2_STANDARD_DIRECT_PROFILE,
    "capability": _LTX2_STANDARD_DIRECT_CAPABILITY,
    "roles": _LTX2_GRAPH_ROLES,
    "edges": _LTX2_GRAPH_EDGES,
    "bindings": _LTX2_GRAPH_BINDINGS,
}

_BUILTIN_IMAGE_OPERATION_MODES = (
    "image_adjustment",
    "image_filter",
    "image_crop",
    "image_upscale",
    "image_stitch",
    "image_tile",
    "image_channels",
    "mask_composite",
)
_BUILTIN_IMAGE_OPERATION_PIPELINE_CLASS = "BuiltinImageOperationV1"
_BUILTIN_IMAGE_OPERATION_REPO = "builtin://modiff/image-operations/v1"
_BUILTIN_IMAGE_OPERATION_PROFILE = {
    "id": "builtin-image-operations:direct",
    "model_type": "BuiltinImageOperation",
    "modes": _BUILTIN_IMAGE_OPERATION_MODES,
    "loader_module": "modules.ImageOperations",
    "loader_action": "ProcessImage",
    "execution_path": "builtin-image-operation",
    "pipeline_class": _BUILTIN_IMAGE_OPERATION_PIPELINE_CLASS,
    "default_repo": _BUILTIN_IMAGE_OPERATION_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": (OFFLOAD_MODE_NONE,),
    "retry_offload_modes": (),
    "max_low_memory_side": 4096,
    "max_low_memory_steps": None,
    "live_proof": False,
    "optional_runtime_profiles": (),
    "optional_runtime_delivery": "base",
    "optional_runtime_platform_deliveries": (),
    "compatible_repos": (),
}
_BUILTIN_IMAGE_OPERATION_CAPABILITY = {
    "modelType": "BuiltinImageOperation",
    "label": "Built-in Image Operations",
    "displayName": "Built-in Image Operations",
    "family": "Built-in Media",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": _BUILTIN_IMAGE_OPERATION_REPO,
    "artifactLabel": "Versioned MoDiff built-in operation contract",
    "artifactKind": "builtin",
    "artifactInstallRequired": False,
    "downloadFiles": [],
    "defaultDtype": "float32",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "source"},
    "recommendedSteps": 1,
    "recommendedGuidance": 0.0,
    "guidanceLabel": "Not used",
    "supportsNegativePrompt": False,
    "supportsImageInput": True,
    "supportsAudioInput": False,
    "supportsMask": True,
    "supportsMultiImage": True,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": False,
    "outputKind": "image",
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
        "width": 1024,
        "height": 1024,
    },
    "modes": list(_BUILTIN_IMAGE_OPERATION_MODES),
    "executionStatus": "supported",
    "revisionCandidates": [],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "liveProof": False,
    "modeRequirements": {
        mode: (
            {
                "requiredImages": ["referenceImages", "maskImage"],
                "minimumCounts": {"referenceImages": 2},
                "note": "Requires two same-size local source images and one same-size mask; no model or network access is used.",
            }
            if mode == "mask_composite"
            else {
                "requiredImages": ["referenceImages"],
                "minimumCounts": {"referenceImages": 2},
                "note": "Requires two to 64 local source images; no model or network access is used.",
            }
            if mode == "image_stitch"
            else {
                "requiredImages": ["referenceImages"],
                "note": "Requires one local source image; no model or network access is used.",
            }
        )
        for mode in _BUILTIN_IMAGE_OPERATION_MODES
    },
    "notes": [
        "Runs deterministic bounded Pillow operations in the base MoDiff process.",
        "No model artifact, optional runtime, accelerator, download, or remote code is required.",
        "Gallery publication remains disabled pending authored examples and human review.",
    ],
}
_BUILTIN_IMAGE_OPERATION_ROLES = (
    ("loadImage", "modules.Image.Load", -620, -80),
    ("imageOperation", "modules.ImageOperations.ProcessImage", -160, -80),
    ("preview", "modules.Image.Preview", 300, -80),
)
_BUILTIN_IMAGE_OPERATION_EDGES = (
    ("loadImage", "image", "imageOperation", "image"),
    ("imageOperation", "output", "preview", "image"),
)
_BUILTIN_IMAGE_OPERATION_BINDINGS = (
    ("loadImage", "file", "referenceImages"),
    ("loadImage", "alpha_channel", "alphaMode"),
    ("imageOperation", "pipeline_class", "pipelineClass"),
    ("imageOperation", "operation", "mode"),
)
_BUILTIN_MASK_COMPOSITE_ROLES = (
    ("loadImage", "modules.Image.Load", -720, -180),
    ("loadMask", "modules.Image.Load", -720, 240),
    ("imageOperation", "modules.ImageOperations.ProcessImage", -160, -80),
    ("preview", "modules.Image.Preview", 400, -80),
)
_BUILTIN_MASK_COMPOSITE_EDGES = (
    ("loadImage", "image", "imageOperation", "image"),
    ("loadMask", "image", "imageOperation", "mask"),
    ("imageOperation", "output", "preview", "image"),
)
_BUILTIN_MASK_COMPOSITE_BINDINGS = _BUILTIN_IMAGE_OPERATION_BINDINGS + (
    ("loadMask", "file", "maskImage"),
    ("loadMask", "alpha_channel", "removeAlpha"),
)
for _builtin_image_mode in _BUILTIN_IMAGE_OPERATION_MODES:
    STUDIO_EXECUTION_SPEC_DEFINITIONS[f"builtin-image-operations:{_builtin_image_mode.replace('_', '-')}:v1"] = {
        "modelType": "BuiltinImageOperation",
        "mode": _builtin_image_mode,
        "profile": _BUILTIN_IMAGE_OPERATION_PROFILE,
        "capability": _BUILTIN_IMAGE_OPERATION_CAPABILITY,
        "roles": (
            _BUILTIN_MASK_COMPOSITE_ROLES
            if _builtin_image_mode == "mask_composite"
            else _BUILTIN_IMAGE_OPERATION_ROLES
        ),
        "edges": (
            _BUILTIN_MASK_COMPOSITE_EDGES
            if _builtin_image_mode == "mask_composite"
            else _BUILTIN_IMAGE_OPERATION_EDGES
        ),
        "bindings": (
            _BUILTIN_MASK_COMPOSITE_BINDINGS
            if _builtin_image_mode == "mask_composite"
            else _BUILTIN_IMAGE_OPERATION_BINDINGS
        ),
    }

_BUILTIN_AUDIO_OPERATION_MODES = ("audio_trim", "audio_join", "audio_loudness_match")
_BUILTIN_AUDIO_OPERATION_PIPELINE_CLASS = "BuiltinAudioOperationV1"
_BUILTIN_AUDIO_OPERATION_REPO = "builtin://modiff/audio-operations/v1"
_BUILTIN_AUDIO_OPERATION_PROFILE = {
    "id": "builtin-audio-operations:direct",
    "model_type": "BuiltinAudioOperation",
    "modes": _BUILTIN_AUDIO_OPERATION_MODES,
    "loader_module": "modules.Audio",
    "loader_action": "ProcessAudio",
    "execution_path": "builtin-audio-operation",
    "pipeline_class": _BUILTIN_AUDIO_OPERATION_PIPELINE_CLASS,
    "default_repo": _BUILTIN_AUDIO_OPERATION_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": (OFFLOAD_MODE_NONE,),
    "retry_offload_modes": (),
    "max_low_memory_side": None,
    "max_low_memory_steps": None,
    "live_proof": False,
    "optional_runtime_profiles": (),
    "optional_runtime_delivery": "base",
    "optional_runtime_platform_deliveries": (),
    "compatible_repos": (),
}
_BUILTIN_AUDIO_OPERATION_CAPABILITY = {
    "modelType": "BuiltinAudioOperation",
    "label": "Built-in Audio Operations",
    "displayName": "Built-in Audio Operations",
    "family": "Built-in Media",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": _BUILTIN_AUDIO_OPERATION_REPO,
    "artifactLabel": "Versioned MoDiff built-in operation contract",
    "artifactKind": "builtin",
    "artifactInstallRequired": False,
    "downloadFiles": [],
    "defaultDtype": "float32",
    "defaultSize": {"width": 0, "height": 0, "aspectRatio": "audio"},
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
    "outputKind": "audio",
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
        "width": 0,
        "height": 0,
    },
    "modes": list(_BUILTIN_AUDIO_OPERATION_MODES),
    "executionStatus": "supported",
    "revisionCandidates": [],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "liveProof": False,
    "modeRequirements": {
        "audio_trim": {
            "requiredAudio": ["sourceAudio"],
            "note": "Requires one bounded local audio source.",
        },
        "audio_join": {
            "requiredAudio": ["sourceAudio", "referenceAudio"],
            "note": "Requires source and continuation audio within the combined sample budget.",
        },
        "audio_loudness_match": {
            "requiredAudio": ["sourceAudio", "referenceAudio"],
            "note": "Requires source and reference audio and applies bounded constant-gain loudness matching.",
        },
    },
    "notes": [
        "Runs bounded in-memory audio operations in the base MoDiff process.",
        "No model artifact, optional runtime, accelerator, download, or remote code is required.",
        "Gallery publication remains disabled pending authored examples and human review.",
    ],
}
_BUILTIN_AUDIO_OPERATION_BINDINGS = (
    ("audioOperation", "pipeline_class", "pipelineClass"),
    ("audioOperation", "operation", "mode"),
    ("loadAudio", "file", "sourceAudio"),
    ("audioExport", "sample_rate", "sampleRate48000"),
)
_BUILTIN_AUDIO_SINGLE_ROLES = (
    ("loadAudio", "modules.Audio.Load", -620, -80),
    ("audioOperation", "modules.Audio.ProcessAudio", -220, -80),
    ("audioExport", "modules.Audio.Export", 260, -80),
)
_BUILTIN_AUDIO_DUAL_ROLES = _BUILTIN_AUDIO_SINGLE_ROLES + (("loadReferenceAudio", "modules.Audio.Load", -620, 160),)
_BUILTIN_AUDIO_SINGLE_EDGES = (
    ("loadAudio", "audio", "audioOperation", "source"),
    ("audioOperation", "output", "audioExport", "audio"),
)
_BUILTIN_AUDIO_DUAL_EDGES = _BUILTIN_AUDIO_SINGLE_EDGES + (
    ("loadReferenceAudio", "audio", "audioOperation", "reference"),
)
for _builtin_audio_mode, _builtin_audio_spec_name in (
    ("audio_trim", "trim"),
    ("audio_join", "join"),
    ("audio_loudness_match", "loudness-match"),
):
    _builtin_audio_dual = _builtin_audio_mode != "audio_trim"
    STUDIO_EXECUTION_SPEC_DEFINITIONS[f"builtin-audio-operations:{_builtin_audio_spec_name}:v1"] = {
        "modelType": "BuiltinAudioOperation",
        "mode": _builtin_audio_mode,
        "profile": _BUILTIN_AUDIO_OPERATION_PROFILE,
        "capability": _BUILTIN_AUDIO_OPERATION_CAPABILITY,
        "roles": _BUILTIN_AUDIO_DUAL_ROLES if _builtin_audio_dual else _BUILTIN_AUDIO_SINGLE_ROLES,
        "edges": _BUILTIN_AUDIO_DUAL_EDGES if _builtin_audio_dual else _BUILTIN_AUDIO_SINGLE_EDGES,
        "bindings": _BUILTIN_AUDIO_OPERATION_BINDINGS + (("loadReferenceAudio", "file", "referenceAudio"),)
        if _builtin_audio_dual
        else _BUILTIN_AUDIO_OPERATION_BINDINGS,
    }

_BUILTIN_DATA_OPERATION_MODES = ("text_select", "data_conversion", "graph_utility")
_BUILTIN_DATA_OPERATION_PIPELINE_CLASS = "BuiltinDataOperationV1"
_BUILTIN_DATA_OPERATION_REPO = "builtin://modiff/data-operations/v1"
_BUILTIN_DATA_OPERATION_PROFILE = {
    "id": "builtin-data-operations:direct",
    "model_type": "BuiltinDataOperation",
    "modes": _BUILTIN_DATA_OPERATION_MODES,
    "loader_module": "modules.Text",
    "loader_action": "ProcessText",
    "execution_path": "builtin-data-operation",
    "pipeline_class": _BUILTIN_DATA_OPERATION_PIPELINE_CLASS,
    "default_repo": _BUILTIN_DATA_OPERATION_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": (OFFLOAD_MODE_NONE,),
    "retry_offload_modes": (),
    "max_low_memory_side": None,
    "max_low_memory_steps": None,
    "live_proof": False,
    "optional_runtime_profiles": (),
    "optional_runtime_delivery": "base",
    "optional_runtime_platform_deliveries": (),
    "compatible_repos": (),
}
_BUILTIN_DATA_OPERATION_CAPABILITY = {
    "modelType": "BuiltinDataOperation",
    "label": "Built-in Data Operations",
    "displayName": "Built-in Data Operations",
    "family": "Built-in Media",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": _BUILTIN_DATA_OPERATION_REPO,
    "artifactLabel": "Versioned MoDiff built-in operation contract",
    "artifactKind": "builtin",
    "artifactInstallRequired": False,
    "downloadFiles": [],
    "defaultDtype": "float32",
    "defaultSize": {"width": 0, "height": 0, "aspectRatio": "data"},
    "recommendedSteps": 1,
    "recommendedGuidance": 0.0,
    "guidanceLabel": "Not used",
    "supportsNegativePrompt": False,
    "supportsImageInput": False,
    "supportsAudioInput": False,
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
        "width": 0,
        "height": 0,
    },
    "modes": list(_BUILTIN_DATA_OPERATION_MODES),
    "executionStatus": "supported",
    "revisionCandidates": [],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "liveProof": False,
    "modeRequirements": {},
    "notes": [
        "Runs deterministic bounded text selection and data conversion in the base MoDiff process.",
        "No model artifact, optional runtime, accelerator, download, or remote code is required.",
        "Gallery publication remains disabled pending authored examples and human review.",
    ],
}
_BUILTIN_DATA_OPERATION_ROLES = (
    ("dataOperation", "modules.Text.ProcessText", -300, -80),
    ("dataPreview", "modules.Primitive.DataViewer", 100, -80),
    ("dataExport", "modules.Primitive.ExportData", 480, -80),
)
_BUILTIN_DATA_OPERATION_EDGES = (
    ("dataOperation", "output", "dataPreview", "value"),
    ("dataPreview", "output", "dataExport", "value"),
)
_BUILTIN_DATA_OPERATION_BINDINGS = (
    ("dataOperation", "pipeline_class", "pipelineClass"),
    ("dataOperation", "operation", "mode"),
    ("dataOperation", "source", "prompt"),
)
for _builtin_data_mode, _builtin_data_spec_name in (
    ("text_select", "text-select"),
    ("data_conversion", "data-conversion"),
    ("graph_utility", "value-switch"),
):
    STUDIO_EXECUTION_SPEC_DEFINITIONS[f"builtin-data-operations:{_builtin_data_spec_name}:v1"] = {
        "modelType": "BuiltinDataOperation",
        "mode": _builtin_data_mode,
        "profile": _BUILTIN_DATA_OPERATION_PROFILE,
        "capability": _BUILTIN_DATA_OPERATION_CAPABILITY,
        "roles": _BUILTIN_DATA_OPERATION_ROLES,
        "edges": _BUILTIN_DATA_OPERATION_EDGES,
        "bindings": _BUILTIN_DATA_OPERATION_BINDINGS,
    }

_BUILTIN_VIDEO_OPERATION_MODES = (
    "video_frame_extract",
    "frame_interpolation",
    "video_stitch",
    "video_trim",
    "video_reverse",
    "video_tile",
)
_BUILTIN_VIDEO_OPERATION_PIPELINE_CLASS = "BuiltinVideoOperationV1"
_BUILTIN_VIDEO_OPERATION_REPO = "builtin://modiff/video-operations/v1"
_BUILTIN_VIDEO_OPERATION_PROFILE = {
    "id": "builtin-video-operations:direct",
    "model_type": "BuiltinVideoOperation",
    "modes": _BUILTIN_VIDEO_OPERATION_MODES,
    "loader_module": "modules.Video",
    "loader_action": "ProcessVideo",
    "execution_path": "builtin-video-operation",
    "pipeline_class": _BUILTIN_VIDEO_OPERATION_PIPELINE_CLASS,
    "default_repo": _BUILTIN_VIDEO_OPERATION_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": (OFFLOAD_MODE_NONE,),
    "retry_offload_modes": (),
    "max_low_memory_side": None,
    "max_low_memory_steps": None,
    "live_proof": False,
    "optional_runtime_profiles": (),
    "optional_runtime_delivery": "base",
    "optional_runtime_platform_deliveries": (),
    "compatible_repos": (),
}
_BUILTIN_VIDEO_OPERATION_CAPABILITY = {
    "modelType": "BuiltinVideoOperation",
    "label": "Built-in Video Operations",
    "displayName": "Built-in Video Operations",
    "family": "Built-in Media",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": _BUILTIN_VIDEO_OPERATION_REPO,
    "artifactLabel": "Versioned MoDiff built-in operation contract",
    "artifactKind": "builtin",
    "artifactInstallRequired": False,
    "downloadFiles": [],
    "defaultDtype": "float32",
    "defaultSize": {"width": 1024, "height": 576, "aspectRatio": "source"},
    "recommendedSteps": 1,
    "recommendedGuidance": 0.0,
    "guidanceLabel": "Not used",
    "supportsNegativePrompt": False,
    "supportsImageInput": False,
    "supportsAudioInput": False,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": False,
    "outputKind": "video",
    "modeOutputKinds": {
        "video_frame_extract": "image",
        "frame_interpolation": "video",
        "video_stitch": "video",
        "video_trim": "video",
        "video_reverse": "video",
        "video_tile": "video",
    },
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
        "width": 1024,
        "height": 576,
    },
    "modes": list(_BUILTIN_VIDEO_OPERATION_MODES),
    "executionStatus": "supported",
    "revisionCandidates": [],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "liveProof": False,
    "modeRequirements": {
        "video_frame_extract": {
            "requiredVideos": ["sourceVideo"],
            "note": "Requires one local source video and extracts at most 64 frames without model or network access.",
        },
        "frame_interpolation": {
            "requiredVideos": ["sourceVideo"],
            "note": (
                "Requires one local source video and performs bounded deterministic FFmpeg blend interpolation; "
                "it is not an optical-flow or model-equivalence claim."
            ),
        },
        "video_stitch": {
            "requiredVideos": ["referenceVideos"],
            "minimumCounts": {"referenceVideos": 2},
            "note": "Requires two to sixteen local source videos and uses the app-owned bounded FFmpeg path.",
        },
        "video_trim": {
            "requiredVideos": ["sourceVideo"],
            "note": "Requires one local source video and applies a finite trim range through app-owned FFmpeg.",
        },
        "video_reverse": {
            "requiredVideos": ["sourceVideo"],
            "note": "Requires one local source video within the explicit reverse pixel-frame memory budget.",
        },
        "video_tile": {
            "requiredVideos": ["referenceVideos"],
            "minimumCounts": {"referenceVideos": 2},
            "note": "Requires two to sixteen local source videos and produces a bounded synchronized tile wall.",
        },
    },
    "notes": [
        "Runs bounded file-backed video operations in the base MoDiff process.",
        "No model artifact, optional runtime, accelerator, download, or remote code is required.",
        "Gallery publication remains disabled pending authored examples and human review.",
    ],
}
_BUILTIN_VIDEO_OPERATION_BINDINGS = (
    ("videoOperation", "pipeline_class", "pipelineClass"),
    ("videoOperation", "operation", "mode"),
)
STUDIO_EXECUTION_SPEC_DEFINITIONS["builtin-video-operations:frame-extract:v1"] = {
    "modelType": "BuiltinVideoOperation",
    "mode": "video_frame_extract",
    "profile": _BUILTIN_VIDEO_OPERATION_PROFILE,
    "capability": _BUILTIN_VIDEO_OPERATION_CAPABILITY,
    "roles": (
        ("videoOperation", "modules.Video.ProcessVideo", -220, -80),
        ("preview", "modules.Image.Preview", 260, -80),
    ),
    "edges": (("videoOperation", "images", "preview", "image"),),
    "bindings": _BUILTIN_VIDEO_OPERATION_BINDINGS + (("videoOperation", "videos", "sourceVideo"),),
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["builtin-video-operations:frame-interpolation:v1"] = {
    "modelType": "BuiltinVideoOperation",
    "mode": "frame_interpolation",
    "profile": _BUILTIN_VIDEO_OPERATION_PROFILE,
    "capability": _BUILTIN_VIDEO_OPERATION_CAPABILITY,
    "roles": (
        ("videoOperation", "modules.Video.ProcessVideo", -220, -80),
        ("videoExport", "modules.Video.Export", 260, -80),
    ),
    "edges": (("videoOperation", "video", "videoExport", "video"),),
    "bindings": _BUILTIN_VIDEO_OPERATION_BINDINGS
    + (
        ("videoOperation", "videos", "sourceVideo"),
        ("videoOperation", "interpolation_fps", "fps"),
        ("videoExport", "fps", "fps"),
    ),
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["builtin-video-operations:stitch:v1"] = {
    "modelType": "BuiltinVideoOperation",
    "mode": "video_stitch",
    "profile": _BUILTIN_VIDEO_OPERATION_PROFILE,
    "capability": _BUILTIN_VIDEO_OPERATION_CAPABILITY,
    "roles": (
        ("videoOperation", "modules.Video.ProcessVideo", -220, -80),
        ("videoExport", "modules.Video.Export", 260, -80),
    ),
    "edges": (("videoOperation", "video", "videoExport", "video"),),
    "bindings": _BUILTIN_VIDEO_OPERATION_BINDINGS
    + (
        ("videoOperation", "videos", "referenceVideos"),
        ("videoExport", "fps", "fps"),
    ),
}
for _builtin_video_mode, _builtin_video_spec_name, _builtin_video_field in (
    ("video_trim", "trim", "sourceVideo"),
    ("video_reverse", "reverse", "sourceVideo"),
    ("video_tile", "tile", "referenceVideos"),
):
    STUDIO_EXECUTION_SPEC_DEFINITIONS[f"builtin-video-operations:{_builtin_video_spec_name}:v1"] = {
        "modelType": "BuiltinVideoOperation",
        "mode": _builtin_video_mode,
        "profile": _BUILTIN_VIDEO_OPERATION_PROFILE,
        "capability": _BUILTIN_VIDEO_OPERATION_CAPABILITY,
        "roles": (
            ("videoOperation", "modules.Video.ProcessVideo", -220, -80),
            ("videoExport", "modules.Video.Export", 260, -80),
        ),
        "edges": (("videoOperation", "video", "videoExport", "video"),),
        "bindings": _BUILTIN_VIDEO_OPERATION_BINDINGS
        + (
            ("videoOperation", "videos", _builtin_video_field),
            ("videoExport", "fps", "fps"),
        ),
    }

_SPANDREL_VIDEO_UPSCALE_MODEL_TYPE = "SpandrelVideoUpscale"
_SPANDREL_VIDEO_UPSCALE_MODE = "video_upscale"
_SPANDREL_VIDEO_UPSCALE_PIPELINE_CLASS = "SpandrelVideoUpscaleV1"
_SPANDREL_VIDEO_UPSCALE_REPO = "nateraw/real-esrgan"
_SPANDREL_VIDEO_UPSCALE_REVISION = require_catalog_revision(_SPANDREL_VIDEO_UPSCALE_REPO)
_SPANDREL_VIDEO_UPSCALE_FILES = ("RealESRGAN_x2plus.pth",)
_SPANDREL_VIDEO_UPSCALE_PROFILE = {
    "id": "real-esrgan-x2-video-upscale:direct",
    "model_type": _SPANDREL_VIDEO_UPSCALE_MODEL_TYPE,
    "modes": (_SPANDREL_VIDEO_UPSCALE_MODE,),
    "loader_module": "modules.Video",
    "loader_action": "UpscaleVideo",
    "execution_path": "spandrel-video-upscale",
    "pipeline_class": _SPANDREL_VIDEO_UPSCALE_PIPELINE_CLASS,
    "default_repo": _SPANDREL_VIDEO_UPSCALE_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": (OFFLOAD_MODE_NONE,),
    "retry_offload_modes": (),
    "max_low_memory_side": 2048,
    "max_low_memory_steps": None,
    "live_proof": True,
    "optional_runtime_profiles": (),
    "optional_runtime_delivery": "base",
    "optional_runtime_platform_deliveries": (),
    "compatible_repos": (),
}
_SPANDREL_VIDEO_UPSCALE_CAPABILITY = {
    "modelType": _SPANDREL_VIDEO_UPSCALE_MODEL_TYPE,
    "label": "Real-ESRGAN x2 Video Upscale",
    "displayName": "Real-ESRGAN x2 Video Upscale",
    "family": "Real-ESRGAN",
    "supportTier": "supported",
    "qualificationStatus": "execution-qualified-gallery-review-pending",
    "qualifiedModes": [_SPANDREL_VIDEO_UPSCALE_MODE],
    "defaultRepo": _SPANDREL_VIDEO_UPSCALE_REPO,
    "downloadFiles": list(_SPANDREL_VIDEO_UPSCALE_FILES),
    "artifactLabel": "Reviewed BSD-3-Clause Real-ESRGAN x2 Spandrel weight",
    "artifactKind": "spandrel_upscaler",
    "artifactInstallRequired": True,
    "defaultDtype": "float32",
    "defaultSize": {"width": 1920, "height": 1080, "aspectRatio": "source"},
    "recommendedSteps": 1,
    "recommendedGuidance": 0.0,
    "guidanceLabel": "Not used",
    "supportsNegativePrompt": False,
    "supportsImageInput": False,
    "supportsAudioInput": False,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": False,
    "outputKind": "video",
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
        "width": 1920,
        "height": 1080,
    },
    "modes": [_SPANDREL_VIDEO_UPSCALE_MODE],
    "executionStatus": "expert_only",
    "revisionCandidates": [_SPANDREL_VIDEO_UPSCALE_REVISION],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "liveProof": True,
    "modeRequirements": {
        _SPANDREL_VIDEO_UPSCALE_MODE: {
            "requiredVideos": ["sourceVideo"],
            "note": "Requires one local source video of at most 1,200 frames and streams it through the exact x2 upscaler.",
        }
    },
    "notes": [
        "The action processes one frame at a time and caps both input and output pixels.",
        "The exact single-file model is rehashed from the app-managed cache before execution.",
        "One exact native-2x ROCm execution preserves 81 frames, 16 fps, and 5.06 seconds with hash-bound quality evidence.",
        "Audio is not preserved; Gallery and Auto remain disabled pending workspace-owner quality and rights approval.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["real-esrgan-x2-video-upscale:v1"] = {
    "modelType": _SPANDREL_VIDEO_UPSCALE_MODEL_TYPE,
    "mode": _SPANDREL_VIDEO_UPSCALE_MODE,
    "profile": _SPANDREL_VIDEO_UPSCALE_PROFILE,
    "capability": _SPANDREL_VIDEO_UPSCALE_CAPABILITY,
    "roles": (
        ("videoUpscaler", "modules.Video.UpscaleVideo", -220, -80),
        ("videoExport", "modules.Video.Export", 300, -80),
    ),
    "edges": (("videoUpscaler", "video_out", "videoExport", "video"),),
    "bindings": (
        ("videoUpscaler", "video", "sourceVideo"),
        ("videoUpscaler", "pipeline_class", "pipelineClass"),
        ("videoUpscaler", "operation", "mode"),
        ("videoUpscaler", "device", "device"),
        ("videoUpscaler", "fps", "fps"),
        ("videoExport", "fps", "fps"),
    ),
}

_SPANDREL_IMAGE_UPSCALE_MODEL_TYPE = "SpandrelImageUpscale"
_SPANDREL_IMAGE_UPSCALE_MODE = "image_upscale"
_SPANDREL_IMAGE_UPSCALE_PIPELINE_CLASS = "SpandrelImageUpscaleV1"
_SPANDREL_IMAGE_UPSCALE_PROFILE = {
    "id": "real-esrgan-x2-image-upscale:direct",
    "model_type": _SPANDREL_IMAGE_UPSCALE_MODEL_TYPE,
    "modes": (_SPANDREL_IMAGE_UPSCALE_MODE,),
    "loader_module": "modules.Spandrel",
    "loader_action": "Upscaler",
    "execution_path": "spandrel-image-upscale",
    "pipeline_class": _SPANDREL_IMAGE_UPSCALE_PIPELINE_CLASS,
    "default_repo": _SPANDREL_VIDEO_UPSCALE_REPO,
    "fallback_repo": None,
    "quantizable_components": (),
    "default_quantized_components": (),
    "supported_offload_modes": (OFFLOAD_MODE_NONE,),
    "retry_offload_modes": (),
    "max_low_memory_side": 2048,
    "max_low_memory_steps": None,
    "live_proof": True,
    "optional_runtime_profiles": (),
    "optional_runtime_delivery": "base",
    "optional_runtime_platform_deliveries": (),
    "compatible_repos": (),
}
_SPANDREL_IMAGE_UPSCALE_CAPABILITY = {
    "modelType": _SPANDREL_IMAGE_UPSCALE_MODEL_TYPE,
    "label": "Real-ESRGAN x2 Image Upscale",
    "displayName": "Real-ESRGAN x2 Image Upscale",
    "family": "Real-ESRGAN",
    "supportTier": "supported",
    "qualificationStatus": "execution-qualified-gallery-review-pending",
    "qualifiedModes": [_SPANDREL_IMAGE_UPSCALE_MODE],
    "defaultRepo": _SPANDREL_VIDEO_UPSCALE_REPO,
    "downloadFiles": list(_SPANDREL_VIDEO_UPSCALE_FILES),
    "artifactLabel": "Reviewed BSD-3-Clause Real-ESRGAN x2 Spandrel weight",
    "artifactKind": "spandrel_upscaler",
    "artifactInstallRequired": True,
    "defaultDtype": "float32",
    "defaultSize": {"width": 2048, "height": 2048, "aspectRatio": "source"},
    "recommendedSteps": 1,
    "recommendedGuidance": 0.0,
    "guidanceLabel": "Not used",
    "supportsNegativePrompt": False,
    "supportsImageInput": True,
    "supportsAudioInput": False,
    "supportsMask": False,
    "supportsMultiImage": False,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": False,
    "outputKind": "image",
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
        "width": 2048,
        "height": 2048,
    },
    "modes": [_SPANDREL_IMAGE_UPSCALE_MODE],
    "executionStatus": "expert_only",
    "revisionCandidates": [_SPANDREL_VIDEO_UPSCALE_REVISION],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "liveProof": True,
    "modeRequirements": {
        _SPANDREL_IMAGE_UPSCALE_MODE: {
            "requiredImages": ["referenceImages"],
            "note": "Requires one local source image and applies the exact cached Real-ESRGAN x2 model with bounded tiled inference.",
        }
    },
    "notes": [
        "Uses the generic Spandrel loader rather than a model-named node.",
        "The exact single-file x2 model is rehashed from the app-managed cache before execution.",
        "Tiled inference preserves overlap context and rejects an invalid or inconsistent model scale.",
        "One exact native-2x ROCm execution is recorded with a hash-bound comparison asset and runtime receipt.",
        "Gallery publication remains disabled pending workspace-owner quality and rights approval.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS["real-esrgan-x2-image-upscale:v1"] = {
    "modelType": _SPANDREL_IMAGE_UPSCALE_MODEL_TYPE,
    "mode": _SPANDREL_IMAGE_UPSCALE_MODE,
    "profile": _SPANDREL_IMAGE_UPSCALE_PROFILE,
    "capability": _SPANDREL_IMAGE_UPSCALE_CAPABILITY,
    "roles": (
        ("loadImage", "modules.Image.Load", -620, -80),
        ("imageUpscaler", "modules.Spandrel.Upscaler", -140, -80),
        ("preview", "modules.Image.Preview", 340, -80),
    ),
    "edges": (
        ("loadImage", "image", "imageUpscaler", "image"),
        ("imageUpscaler", "output", "preview", "image"),
    ),
    "bindings": (
        ("loadImage", "file", "referenceImages"),
        ("loadImage", "alpha_channel", "alphaMode"),
        ("imageUpscaler", "device", "device"),
    ),
}

_FLUX2_DEV_DIRECT_PROFILE = {
    "id": "flux2-dev:direct",
    "model_type": "Flux2Pipeline",
    "modes": ("text_to_image", "multi_image_reference_edit"),
    "loader_module": "modules.DiffusersImage",
    "loader_action": "LoadPipeline",
    "execution_path": "direct-diffusers-image",
    "pipeline_class": "Flux2Pipeline",
    "default_repo": FLUX2_DEV_REPO,
    "fallback_repo": None,
    "quantizable_components": ("transformer", "text_encoder"),
    "default_quantized_components": ("transformer", "text_encoder"),
    "supported_offload_modes": (
        OFFLOAD_MODE_MODEL_CPU,
        OFFLOAD_MODE_SEQUENTIAL_CPU,
        OFFLOAD_MODE_GROUP_CPU,
        OFFLOAD_MODE_GROUP_DISK,
        OFFLOAD_MODE_NONE,
    ),
    "retry_offload_modes": (OFFLOAD_MODE_GROUP_DISK,),
    "max_low_memory_side": 768,
    "max_low_memory_steps": 20,
    "live_proof": False,
    "compatible_repos": (),
}
_FLUX2_DEV_DIRECT_CAPABILITY = {
    "modelType": "Flux2Pipeline",
    "label": "FLUX.2 dev",
    "displayName": "FLUX.2-dev",
    "family": "FLUX Image",
    "supportTier": "supported",
    "qualificationStatus": "graph-qualified-execution-pending",
    "qualifiedModes": [],
    "defaultRepo": FLUX2_DEV_REPO,
    "downloadFiles": FLUX2_DEV_DIFFUSERS_FILES,
    "artifactLabel": "Pinned gated FLUX.2 Diffusers repository",
    "defaultDtype": "bfloat16",
    "defaultSize": {"width": 1024, "height": 1024, "aspectRatio": "1:1"},
    "recommendedSteps": 50,
    "recommendedGuidance": 4.0,
    "guidanceLabel": "Guidance",
    "supportsNegativePrompt": False,
    "supportsImageInput": True,
    "supportsMask": False,
    "supportsMultiImage": True,
    "supportsControlImage": False,
    "supportsLayers": False,
    "supportsLora": True,
    "outputKind": "image",
    "offloadSupport": {
        "default": OFFLOAD_MODE_GROUP_DISK,
        "lowVram": OFFLOAD_MODE_GROUP_DISK,
        "emergency": OFFLOAD_MODE_GROUP_DISK,
        "modes": list(_FLUX2_DEV_DIRECT_PROFILE["supported_offload_modes"]),
    },
    "lowVram": {
        "dtype": "bfloat16",
        "autoOffload": True,
        "offloadMode": OFFLOAD_MODE_GROUP_DISK,
        "steps": 20,
        "width": 768,
        "height": 768,
    },
    "modes": list(_FLUX2_DEV_DIRECT_PROFILE["modes"]),
    "modeRequirements": {
        "multi_image_reference_edit": {
            "requiredImages": ["referenceImages"],
            "note": "Accepts one or more ordered reference images through the official FLUX.2 image-conditioned call.",
        }
    },
    "executionStatus": "expert_only",
    "revisionCandidates": [
        require_catalog_revision(FLUX2_DEV_REPO, model_type="Flux2ModularPipeline")
    ],
    "autoEligible": False,
    "templateEligible": True,
    "galleryEligible": False,
    "liveProof": False,
    "notes": [
        "The bounded Diffusers component closure is 112,823,045,100 bytes and remains gated by the FLUX non-commercial license.",
        "Public execution remains disabled until exact license acceptance, installation, lifecycle evidence, and output review are recorded.",
    ],
}
STUDIO_EXECUTION_SPEC_DEFINITIONS.update(
    {
        "flux2-dev:text-to-image:v1": {
            "modelType": "Flux2Pipeline",
            "mode": "text_to_image",
            "profile": _FLUX2_DEV_DIRECT_PROFILE,
            "capability": _FLUX2_DEV_DIRECT_CAPABILITY,
            "roles": _GRAPH_ROLES,
            "edges": _GRAPH_EDGES,
            "bindings": _GRAPH_BINDINGS,
        },
        "flux2-dev:multi-image-reference-edit:v1": {
            "modelType": "Flux2Pipeline",
            "mode": "multi_image_reference_edit",
            "profile": _FLUX2_DEV_DIRECT_PROFILE,
            "capability": _FLUX2_DEV_DIRECT_CAPABILITY,
            "roles": _EDIT_GRAPH_ROLES,
            "edges": _EDIT_GRAPH_EDGES,
            "bindings": _EDIT_GRAPH_BINDINGS,
        },
    }
)

_EXPERT_IMAGE_QUANTIZATION_PROFILE_IDS = {
    "flux-canny:direct",
    "flux-canny:img2img-direct",
    "flux-canny:inpaint-direct",
    "flux-depth:direct",
    "flux-depth:img2img-direct",
    "flux-depth:inpaint-direct",
    "flux-dev:direct",
    "flux-dev:img2img-direct",
    "flux-dev:inpaint-direct",
    "flux-dev:modular",
    "flux-kontext:modular",
    "flux-fill:direct",
    "flux-kontext:direct",
    "flux-kontext-inpaint:direct",
    "flux-krea:direct",
    "flux-redux:direct",
    "flux-schnell:direct",
    "flux2-klein:direct",
    "flux2-dev:direct",
    "flux2-modular:equivalent-standard",
    "flux2-klein:modular",
    "flux2-klein-base:modular",
    "flux2-klein-inpaint:direct",
}
def _equivalent_standard_profile(
    source_spec_id: str,
    *,
    profile_id: str,
    model_type: str,
    modes: tuple[str, ...],
) -> dict[str, Any]:
    profile = deepcopy(STUDIO_EXECUTION_SPEC_DEFINITIONS[source_spec_id]["profile"])
    profile.update({"id": profile_id, "model_type": model_type, "modes": modes, "live_proof": False})
    return profile


def _equivalent_standard_capability(
    source_spec_id: str,
    *,
    model_type: str,
    label: str,
    modes: tuple[str, ...],
) -> dict[str, Any]:
    capability = deepcopy(STUDIO_EXECUTION_SPEC_DEFINITIONS[source_spec_id]["capability"])
    requirements = capability.get("modeRequirements") or {}
    capability.update(
        {
            "modelType": model_type,
            "label": label,
            "displayName": label,
            "qualificationStatus": "graph-qualified-execution-pending",
            "qualifiedModes": [],
            "modes": list(modes),
            "modeRequirements": {mode: deepcopy(requirements.get(mode, {})) for mode in modes},
            "executionStatus": "expert_only",
            "autoEligible": False,
            "galleryEligible": False,
            "liveProof": False,
        }
    )
    capability.setdefault("notes", []).append(
        "This Cluster retains the exact Modular Diffusers block hierarchy while executing through the reviewed "
        "equivalent standard Diffusers pipeline until a split-block runtime is separately qualified."
    )
    return capability


def _with_exact_video_revision(bindings: tuple) -> tuple:
    return tuple(
        (role, param, "defaultRevision") if role == "wanPipeline" and param == "revision" else (role, param, source)
        for role, param, source in bindings
    )


_LTX_EQUIVALENT_CAPABILITY = _planning_video_capability(
    "LTXModularPipeline",
    "LTX Modular — Equivalent Standard Execution",
    "LTX Video",
    LTX_VIDEO_REPO,
    ("text_to_video", "image_to_video"),
    {
        "image_to_video": {
            "requiredImages": ["referenceImages"],
            "note": "Requires one source image for the selected upstream image2video workflow.",
        }
    },
    download_files=LTX_VIDEO_DIFFUSERS_FILES,
)
_LTX_EQUIVALENT_CAPABILITY.update(
    {
        "qualificationStatus": "graph-qualified-execution-pending",
        "qualifiedModes": [],
        "executionStatus": "expert_only",
        "autoEligible": False,
        "galleryEligible": False,
        "liveProof": False,
    }
)
_LTX_EQUIVALENT_CAPABILITY.setdefault("notes", []).append(
    "The Cluster preserves the upstream LTX Modular block hierarchy and uses the reviewed LTXConditionPipeline "
    "as its explicitly equivalent full-pipeline executor."
)

_LTX2_EQUIVALENT_CAPABILITY = _planning_video_capability(
    "LTX2ModularPipeline",
    "LTX-2 Modular — Equivalent Standard Execution",
    "LTX Video",
    LTX2_REPO,
    ("text_to_video", "image_to_video", "reference_to_video", "in_context_to_video"),
    {
        "image_to_video": {
            "requiredImages": ["referenceImages"],
            "note": "Requires one source image for the selected upstream image2video workflow.",
        },
        "reference_to_video": {
            "requiredImages": ["conditionImages"],
            "note": (
                "Requires one or more ordered condition images. The execution adapter converts them to official "
                "LTX2VideoCondition instances and distributes them from the first through last generated frame."
            ),
        },
        "in_context_to_video": {
            "requiredVideos": ["referenceVideos"],
            "note": (
                "Requires one reference video, the exact official Canny preprocessing graph, and the pinned "
                "LTX-2 IC-LoRA dependency."
            ),
        },
    },
    download_files=LTX2_DIFFUSERS_FILES,
)
_LTX2_EQUIVALENT_CAPABILITY.update(
    {
        "qualificationStatus": "graph-qualified-execution-pending",
        "qualifiedModes": [],
        "executionStatus": "expert_only",
        "outputMedia": ["video", "audio"],
        "autoEligible": False,
        "galleryEligible": False,
        "liveProof": False,
    }
)
_LTX2_EQUIVALENT_CAPABILITY.setdefault("notes", []).append(
    "The upstream text2video, image2video, ordered-image condition, and in-context workflows retain their exact "
    "Modular block hierarchies. In-context uses its distinct LTX2InContextPipeline and pinned IC-LoRA executor."
)

_EQUIVALENT_STANDARD_STUDIO_SPECS = {
    "flux2-modular:equivalent-standard-text-to-image:v1": {
        "source": "flux2-dev:text-to-image:v1",
        "modelType": "Flux2ModularPipeline",
        "mode": "text_to_image",
        "profileId": "flux2-modular:equivalent-standard",
        "modes": ("text_to_image", "multi_image_reference_edit"),
        "label": "FLUX.2 Modular — Equivalent Standard Execution",
    },
    "flux2-modular:equivalent-standard-image-conditioned:v1": {
        "source": "flux2-dev:multi-image-reference-edit:v1",
        "modelType": "Flux2ModularPipeline",
        "mode": "multi_image_reference_edit",
        "profileId": "flux2-modular:equivalent-standard",
        "modes": ("text_to_image", "multi_image_reference_edit"),
        "label": "FLUX.2 Modular — Equivalent Standard Execution",
    },
    "ernie-image:equivalent-standard-text-to-image:v1": {
        "source": "ernie-image-turbo:text-to-image:v1",
        "modelType": "ErnieImageModularPipeline",
        "mode": "text_to_image",
        "profileId": "ernie-image:equivalent-standard",
        "modes": ("text_to_image",),
        "label": "ERNIE Image Modular — Equivalent Standard Execution",
    },
    "ltx:equivalent-standard-text-to-video:v1": {
        "source": "ltx-video-0.9.8-13b-distilled:text-to-video:v1",
        "modelType": "LTXModularPipeline",
        "mode": "text_to_video",
        "profileId": "ltx:equivalent-standard",
        "modes": ("text_to_video", "image_to_video"),
    },
    "ltx:equivalent-standard-image-to-video:v1": {
        "source": "ltx-video-0.9.8-13b-distilled:image-to-video:v1",
        "modelType": "LTXModularPipeline",
        "mode": "image_to_video",
        "profileId": "ltx:equivalent-standard",
        "modes": ("text_to_video", "image_to_video"),
    },
    "wan22:equivalent-standard-text-to-video:v1": {
        "source": "wan-22-a14b:text-to-video:v1",
        "modelType": "Wan22ModularPipeline",
        "mode": "text_to_video",
        "profileId": "wan22:equivalent-standard",
        "modes": ("text_to_video",),
        "label": "Wan 2.2 Modular — Equivalent Standard Execution",
    },
    "wan22-i2v:equivalent-standard-image-to-video:v1": {
        "source": "wan-22-i2v-a14b:image-to-video:v1",
        "modelType": "Wan22Image2VideoModularPipeline",
        "mode": "image_to_video",
        "profileId": "wan22-i2v:equivalent-standard",
        "modes": ("image_to_video",),
        "label": "Wan 2.2 Image-to-Video Modular — Equivalent Standard Execution",
    },
    "ltx2-modular:equivalent-standard-text-to-video:v1": {
        "source": "ltx2:text-to-video:v1",
        "modelType": "LTX2ModularPipeline",
        "mode": "text_to_video",
        "profileId": "ltx2-modular:equivalent-standard",
        "modes": ("text_to_video", "image_to_video", "reference_to_video"),
    },
    "ltx2-modular:equivalent-standard-image-to-video:v1": {
        "source": "ltx2:image-to-video:v1",
        "modelType": "LTX2ModularPipeline",
        "mode": "image_to_video",
        "profileId": "ltx2-modular:equivalent-standard",
        "modes": ("text_to_video", "image_to_video", "reference_to_video"),
    },
    "ltx2-modular:equivalent-standard-condition-images:v1": {
        "source": "ltx2:reference-to-video:v1",
        "modelType": "LTX2ModularPipeline",
        "mode": "reference_to_video",
        "profileId": "ltx2-modular:equivalent-standard",
        "modes": ("text_to_video", "image_to_video", "reference_to_video"),
        "bindingSourceAliases": {"referenceImages": "conditionImages"},
    },
    "ltx2-modular:equivalent-standard-in-context-canny:v1": {
        "source": "ltx2:in-context-reference-to-video:v1",
        "modelType": "LTX2ModularPipeline",
        "mode": "in_context_to_video",
        "profileId": "ltx2-modular:equivalent-in-context",
        "modes": ("in_context_to_video",),
    },
}
for _spec_id, _equivalent in _EQUIVALENT_STANDARD_STUDIO_SPECS.items():
    _source_definition = STUDIO_EXECUTION_SPEC_DEFINITIONS[_equivalent["source"]]
    _loader_node_key = (
        f"{_source_definition['profile']['loader_module']}.{_source_definition['profile']['loader_action']}"
    )
    _loader_roles = tuple(
        role for role, node_key, _x, _y in _source_definition["roles"] if node_key == _loader_node_key
    )
    if len(_loader_roles) != 1:
        raise ValueError(f"Equivalent Studio execution spec {_spec_id!r} has no unique loader role.")
    _binding_source_aliases = _equivalent.get("bindingSourceAliases", {})
    _equivalent_definition = {
        "modelType": _equivalent["modelType"],
        "mode": _equivalent["mode"],
        "profile": _equivalent_standard_profile(
            _equivalent["source"],
            profile_id=_equivalent["profileId"],
            model_type=_equivalent["modelType"],
            modes=_equivalent["modes"],
        ),
        "roles": deepcopy(_source_definition["roles"]),
        "edges": deepcopy(_source_definition["edges"]),
        "bindings": tuple(
            (role, field, _binding_source_aliases.get(source, source))
            for role, field, source in _with_exact_video_revision(tuple(_source_definition["bindings"]))
        )
        + ((_loader_roles[0], "execution_profile_id", "executionProfileId"),),
    }
    if "expertResourceRequirements" in _source_definition:
        _equivalent_definition["expertResourceRequirements"] = deepcopy(
            _source_definition["expertResourceRequirements"]
        )
    if _equivalent["modelType"] == "LTXModularPipeline":
        _equivalent_definition["capability"] = deepcopy(_LTX_EQUIVALENT_CAPABILITY)
    elif _equivalent["modelType"] == "LTX2ModularPipeline":
        _equivalent_definition["capability"] = deepcopy(_LTX2_EQUIVALENT_CAPABILITY)
    elif "label" in _equivalent:
        _equivalent_definition["capability"] = _equivalent_standard_capability(
            _equivalent["source"],
            model_type=_equivalent["modelType"],
            label=_equivalent["label"],
            modes=_equivalent["modes"],
        )
    STUDIO_EXECUTION_SPEC_DEFINITIONS[_spec_id] = _equivalent_definition


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
    if "auxiliaryTerminalRoles" in definition:
        payload["auxiliaryTerminalRoles"] = tuple(definition["auxiliaryTerminalRoles"])
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


def studio_expert_resource_requirements(
    model_type: str,
    mode: str,
    offload_mode: str,
) -> dict[str, Any] | None:
    """Return a reviewed hard resource floor for one explicit Expert recipe.

    Most execution profiles have no hard Expert floor beyond their ordinary
    Auto recommendations. Large dual-expert routes can declare one after a
    measured failure so the same unsafe recipe is rejected before model load.
    """

    matches = [
        definition["expertResourceRequirements"].get(offload_mode)
        for definition in STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
        if definition["modelType"] == model_type
        and definition["mode"] == mode
        and isinstance(definition.get("expertResourceRequirements"), dict)
    ]
    matches = [requirement for requirement in matches if isinstance(requirement, dict)]
    return deepcopy(matches[0]) if len(matches) == 1 else None


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
    "modules.ModularDiffusers.IPAdapter": "ip_adapter",
}


def _execution_spec_role_params(
    public: dict[str, Any],
    node_key: str,
    node: dict[str, Any],
) -> dict[str, Any]:
    params = deepcopy(node["params"])
    if node_key == "modules.ModularDiffusers.ModelsLoader":
        loader_roles = {
            role
            for role, candidate_key, _x, _y in public["roles"]
            if candidate_key == node_key
        }
        connected_optional_inputs = {
            target_handle
            for _source_role, _source_handle, target_role, target_handle in public["edges"]
            if target_role in loader_roles
        }
        # Optional component override sockets are part of the ordinary loader
        # node, but an exact registered Block exposes them only when its
        # reviewed Studio graph actually connects that component. This avoids
        # changing every unrelated immutable BlockDefinitionV2 when support
        # for a new external component is added to ModelsLoader.
        for field in ("controlnet",):
            if field not in connected_optional_inputs:
                params.pop(field, None)
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
            raise ValueError(
                "Studio execution specifications must have unique pairs and consistent execution profiles."
            )
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
            source_types = _param_types(source_param) if isinstance(source_param, dict) else set()
            target_types = _param_types(target_param) if isinstance(target_param, dict) else set()
            if (
                not isinstance(source_param, dict)
                or source_param.get("display") != "output"
                or not isinstance(target_param, dict)
                or target_param.get("display") != "input"
                or not (source_types & target_types or "any" in source_types or "any" in target_types)
            ):
                raise ValueError("Studio execution specification references an incompatible handle.")
            adjacency[source_role].add(target_role)
            adjacency[target_role].add(source_role)
        auxiliary_terminal_roles = public.get("auxiliaryTerminalRoles", ())
        outgoing_roles = {source_role for source_role, _source_handle, _target_role, _target_handle in connections}
        if (
            not isinstance(auxiliary_terminal_roles, (list, tuple))
            or any(not isinstance(role, str) or not role for role in auxiliary_terminal_roles)
            or len(auxiliary_terminal_roles) != len(set(auxiliary_terminal_roles))
            or any(role not in roles or role in outgoing_roles for role in auxiliary_terminal_roles)
        ):
            raise ValueError("Studio execution specification auxiliary terminal roles are invalid.")
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
    candidate_contract = candidate.get("studioExecutionSpecContract") if isinstance(candidate, dict) else None
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
    executable_ids = {str(node_id) for path in paths if isinstance(path, list) for node_id in path}
    controlled_contracts = runtime_hints.get("controlledGraphContracts")
    controlled_contract_ids = (
        tuple(controlled_contracts)
        if isinstance(controlled_contracts, list)
        and len(controlled_contracts) <= 16
        and len(set(controlled_contracts)) == len(controlled_contracts)
        and all(isinstance(item, str) and item for item in controlled_contracts)
        else ()
    )
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
        direct_edge_matches = (
            isinstance(param, dict) and param.get("sourceId") == source_id and param.get("sourceKey") == source_handle
        )
        controlled_video_upscale_matches = False
        if not direct_edge_matches and "upscale.video.v1" in controlled_contract_ids and isinstance(param, dict):
            upscalers = [
                (node_id, node)
                for node_id, node in nodes.items()
                if str(node_id) in executable_ids
                and isinstance(node, dict)
                and node.get("module") == "modules.Spandrel"
                and node.get("action") == "Upscaler"
                and isinstance((node.get("params") or {}).get("image"), dict)
                and (node.get("params") or {})["image"].get("sourceId") == source_id
                and (node.get("params") or {})["image"].get("sourceKey") == source_handle
            ]
            controlled_video_upscale_matches = (
                len(upscalers) == 1
                and param.get("sourceId") == str(upscalers[0][0])
                and param.get("sourceKey") == "output"
            )
        controlled_lora_matches = False
        controlled_lora_module = next(
            (
                module_name
                for contract_id, module_name in (
                    ("lora.diffusers-image.v1", "modules.DiffusersImage"),
                    ("lora.diffusers-audio.v1", "modules.DiffusersAudio"),
                )
                if contract_id in controlled_contract_ids
            ),
            None,
        )
        if not direct_edge_matches and controlled_lora_module and isinstance(param, dict):
            adapters = [
                (node_id, node)
                for node_id, node in nodes.items()
                if str(node_id) in executable_ids
                and isinstance(node, dict)
                and node.get("module") == controlled_lora_module
                and node.get("action") == "LoadAdapter"
                and isinstance((node.get("params") or {}).get("pipeline"), dict)
                and (node.get("params") or {})["pipeline"].get("sourceId") == source_id
                and (node.get("params") or {})["pipeline"].get("sourceKey") == source_handle
            ]
            all_adapters = {
                str(node_id): node
                for node_id, node in nodes.items()
                if str(node_id) in executable_ids
                and isinstance(node, dict)
                and node.get("module") == controlled_lora_module
                and node.get("action") == "LoadAdapter"
                and isinstance((node.get("params") or {}).get("pipeline"), dict)
            }
            if len(adapters) == 1 and 1 <= len(all_adapters) <= 8:
                current_id = str(adapters[0][0])
                visited = {current_id}
                while len(visited) < len(all_adapters):
                    next_adapters = [
                        adapter_id
                        for adapter_id, adapter in all_adapters.items()
                        if adapter_id not in visited
                        and (adapter.get("params") or {})["pipeline"].get("sourceId") == current_id
                        and (adapter.get("params") or {})["pipeline"].get("sourceKey") == "output"
                    ]
                    if len(next_adapters) != 1:
                        break
                    current_id = next_adapters[0]
                    visited.add(current_id)
                controlled_lora_matches = (
                    len(visited) == len(all_adapters)
                    and param.get("sourceId") == current_id
                    and param.get("sourceKey") == "output"
                )
        if not direct_edge_matches and not controlled_video_upscale_matches and not controlled_lora_matches:
            raise RuntimeError("Studio execution specification edge does not match the executable graph.")
    for role, param, _source in spec["bindings"]:
        node = nodes[node_ids[role]]
        if param not in (node.get("params") or {}):
            raise RuntimeError("Studio execution specification binding is missing from the executable graph.")
