import hashlib
import inspect
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter

from modiff.NodeBase import NodeBase
from modiff.diffusers_offload import (
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
    apply_pipeline_offload,
    normalize_offload_mode,
    offload_mode_param,
)
from modiff.model_artifact_catalog import (
    IMMUTABLE_HUB_REVISION,
    catalog_repository_pin,
    catalog_revision,
    require_catalog_revision,
)
from utils.huggingface import (
    cached_file_path,
    local_files_only,
    resolve_managed_hf_cache_file,
    validate_hf_repo_id,
)
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST, str_to_dtype

logger = logging.getLogger("modiff")

FLUX_SCHNELL_REPO = "black-forest-labs/FLUX.1-schnell"
FLUX_DEV_REPO = "black-forest-labs/FLUX.1-dev"
FLUX_DEV_FP8_REPO = "black-forest-labs/FLUX.1-dev-FP8"
FLUX_KREA_REPO = "black-forest-labs/FLUX.1-Krea-dev"
FLUX_KONTEXT_REPO = "black-forest-labs/FLUX.1-Kontext-dev"
FLUX_KONTEXT_NVFP4_REPO = "black-forest-labs/FLUX.1-Kontext-dev-NVFP4"
FLUX_FILL_REPO = "black-forest-labs/FLUX.1-Fill-dev"
FLUX_DEPTH_REPO = "black-forest-labs/FLUX.1-Depth-dev"
FLUX_CANNY_REPO = "black-forest-labs/FLUX.1-Canny-dev"
FLUX_CANNY_REPAIR_REPO = "fuliucansheng/FLUX.1-Canny-dev-diffusers"
FLUX_REDUX_REPO = "black-forest-labs/FLUX.1-Redux-dev"
FLUX2_KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
Z_IMAGE_REPO = "Tongyi-MAI/Z-Image-Turbo"
SDXL_BASE_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_TURBO_REPO = "stabilityai/sdxl-turbo"
SDXL_INSTRUCT_PIX2PIX_REPO = "diffusers/sdxl-instructpix2pix-768"
SDXL_CONTROLNET_CANNY_REPO = "diffusers/controlnet-canny-sdxl-1.0"
SDXL_T2I_ADAPTER_CANNY_REPO = "TencentARC/t2i-adapter-canny-sdxl-1.0"
HUNYUAN_DIT_DISTILLED_REPO = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled"
HUNYUAN_DIT_CONTROLNET_CANNY_REPO = "Tencent-Hunyuan/HunyuanDiT-v1.2-ControlNet-Diffusers-Canny"
SD15_BASE_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD15_CONTROLNET_CANNY_REPO = "lllyasviel/control_v11p_sd15_canny"
SANA_REPO = "Efficient-Large-Model/Sana_600M_1024px_diffusers"
SANA_SPRINT_REPO = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
PIXART_SIGMA_REPO = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
AURAFLOW_V03_REPO = "fal/AuraFlow-v0.3"
CHROMA1_HD_REPO = "lodestones/Chroma1-HD"
COGVIEW3_PLUS_REPO = "zai-org/CogView3-Plus-3B"
COGVIEW4_6B_REPO = "zai-org/CogView4-6B"
ERNIE_IMAGE_TURBO_REPO = "baidu/ERNIE-Image-Turbo"
DREAMLITE_BASE_REPO = "carlofkl/DreamLite-base"
DREAMLITE_MOBILE_REPO = "carlofkl/DreamLite-mobile"
LCM_DREAMSHAPER_REPO = "SimianLuo/LCM_Dreamshaper_v7"
MARIGOLD_DEPTH_LCM_REPO = "prs-eth/marigold-depth-lcm-v1-0"
QWEN_IMAGE_2512_REPO = "Qwen/Qwen-Image-2512"
QWEN_IMAGE_2512_PREQUANTIZED_REPO = "unsloth/Qwen-Image-2512-unsloth-bnb-4bit"
QWEN_IMAGE_EDIT_REPO = "Qwen/Qwen-Image-Edit"
QWEN_IMAGE_EDIT_PLUS_REPO = "Qwen/Qwen-Image-Edit-2511"
QWEN_IMAGE_EDIT_PREQUANTIZED_REPO = "ovedrive/qwen-image-edit-4bit"
DDPM_CIFAR10_REPO = "google/ddpm-cifar10-32"
CONSISTENCY_IMAGENET64_REPO = "openai/diffusers-cd_imagenet64_l2"
DEVICE_OPTIONS = list(DEVICE_LIST.keys())
_IMAGE_MODE_ORDER = (
    "unconditional_image",
    "depth_estimation",
    "text_to_image",
    "edit_image",
    "multi_image_reference_edit",
    "inpaint",
    "outpaint",
    "control_image",
)
_IMAGE_MODEL_SOURCES = frozenset({"hub", "local"})
_MAX_IMAGE_INPUT_DIMENSION = 8192
_MAX_IMAGE_INPUT_PIXELS = 16 * 1024 * 1024
_MAX_IMAGE_OUTPUT_PIXELS = 2048 * 2048


@dataclass(frozen=True)
class ImagePipelineAdapter:
    pipeline_class: str
    modes: frozenset[str]
    default_repo: str
    compatible_repos: frozenset[str] = frozenset()
    artifact_pipeline_classes: tuple[str, ...] = ()
    runtime_pipeline_classes: tuple[str, ...] = ()
    upstream_pipeline_class: str | None = None
    safe_serialization_required: bool = False
    weight_variant: str | None = None
    component_dtype_overrides: tuple[tuple[str, str], ...] = ()
    max_inference_steps: int = 100
    min_output_side: int = 16
    max_output_side: int = 2048
    output_side_step: int = 16
    max_output_pixels: int = _MAX_IMAGE_OUTPUT_PIXELS
    fixed_guidance_scale: float | None = None
    minimum_image_guidance_scale: float = 0.0
    guidance_parameter: str | None = "guidance_scale"
    ignored_generation_parameters: frozenset[str] = frozenset()
    multi_image_strategy: str = "list"
    max_sequence_length: int = 512
    max_reference_images: int = 1
    max_reference_pixels: int = _MAX_IMAGE_INPUT_PIXELS
    unconditional_optional_fields: tuple[str, ...] = ()
    conditioning_kind: str | None = None
    default_conditioning_repo: str | None = None
    conditioning_component_class: str | None = None
    conditioning_component_parameter: str | None = None
    conditioning_weight_variant: str | None = None
    control_image_parameter: str = "control_image"
    conditioning_scale_parameter: str | None = None

    def __post_init__(self) -> None:
        if self.upstream_pipeline_class is not None and not self.upstream_pipeline_class:
            raise ValueError("An upstream image pipeline class cannot be blank.")
        if not 1 <= self.max_inference_steps <= 100:
            raise ValueError("Image adapters must bound inference steps between 1 and 100.")
        if not 16 <= self.min_output_side <= self.max_output_side <= 2048:
            raise ValueError("Image adapters must bound output sides between 16 and 2048.")
        if self.output_side_step not in {16, 32}:
            raise ValueError("Image adapter output-side increments must be 16 or 32 pixels.")
        if self.min_output_side % self.output_side_step or (
            self.max_output_side - self.min_output_side
        ) % self.output_side_step:
            raise ValueError("Image adapter output-side bounds must align to their declared increment.")
        if not self.min_output_side**2 <= self.max_output_pixels <= _MAX_IMAGE_OUTPUT_PIXELS:
            raise ValueError("Image adapters must declare a bounded output-pixel ceiling covering the minimum size.")
        if self.fixed_guidance_scale is not None and not 0.0 <= self.fixed_guidance_scale <= 20.0:
            raise ValueError("An exact text guidance scale must be between 0 and 20.")
        if not 0.0 <= self.minimum_image_guidance_scale <= 20.0:
            raise ValueError("The minimum image guidance scale must be between 0 and 20.")
        if self.guidance_parameter is not None and not self.guidance_parameter:
            raise ValueError("An image guidance parameter cannot be blank.")
        unsupported_ignored = self.ignored_generation_parameters - {"image_guidance_scale"}
        if unsupported_ignored:
            raise ValueError("Image adapters can ignore only reviewed no-op generation parameters.")
        if self.weight_variant is not None and self.weight_variant != "fp16":
            raise ValueError("Only the reviewed fp16 image weight variant is supported.")
        component_names = [name for name, _dtype in self.component_dtype_overrides]
        if len(component_names) != len(set(component_names)) or any(not name for name in component_names):
            raise ValueError("Image component dtype overrides must name unique non-empty components.")
        if any(dtype not in {"float32", "float16", "bfloat16"} for _name, dtype in self.component_dtype_overrides):
            raise ValueError("Image component dtype overrides must use a supported torch dtype.")
        conditioning_fields = (
            self.conditioning_kind,
            self.default_conditioning_repo,
            self.conditioning_component_class,
            self.conditioning_component_parameter,
        )
        if any(value is not None for value in conditioning_fields) and not all(
            isinstance(value, str) and value for value in conditioning_fields
        ):
            raise ValueError("Conditioned image adapters must declare one complete auxiliary component contract.")
        if self.conditioning_weight_variant is not None:
            if self.conditioning_kind is None:
                raise ValueError("A conditioning weight variant requires an auxiliary component contract.")
            if self.conditioning_weight_variant != "fp16":
                raise ValueError("Only the reviewed fp16 conditioning weight variant is supported.")

    @property
    def managed_repos(self) -> frozenset[str]:
        """Reviewed repositories that remain valid when this adapter is selected."""

        return frozenset((self.default_repo, *self.compatible_repos))

    @property
    def model_filter_classes(self) -> tuple[str, ...]:
        return self.artifact_pipeline_classes or (self.pipeline_class,)

    @property
    def load_pipeline_class(self) -> str:
        return self.upstream_pipeline_class or self.pipeline_class

    @property
    def allowed_runtime_classes(self) -> tuple[str, ...]:
        return self.runtime_pipeline_classes or (self.pipeline_class,)

    @property
    def mode_options(self) -> tuple[str, ...]:
        """Expose stable UI/default ordering while preserving the set-valued contract."""

        return tuple(mode for mode in _IMAGE_MODE_ORDER if mode in self.modes)

    def apply_generation_parameters(self, pipeline: Any, values: dict[str, Any], target: dict[str, Any]) -> None:
        aliases = {
            "negative_prompt": "negative_prompt",
            "width": "width",
            "height": "height",
            "max_sequence_length": "max_sequence_length",
            "strength": "strength",
            "padding_mask_crop": "padding_mask_crop",
            "reference_strength": "reference_strength",
            "pag_scale": "pag_scale",
            "pag_adaptive_scale": "pag_adaptive_scale",
            "image_guidance_scale": "image_guidance_scale",
        }
        for source, destination in aliases.items():
            if source in self.ignored_generation_parameters:
                continue
            value = values.get(source)
            # The graph uses zero to mean "no crop". Diffusers uses None for
            # that contract; passing 0 enters Qwen's overlay path and can make
            # its internally resized image conflict with the original-size
            # mask during wide outpaint finalization.
            if source == "padding_mask_crop" and (value is None or int(value) <= 0):
                continue
            if supports_arg(pipeline, destination) and value is not None:
                target[destination] = value
        if (
            self.guidance_parameter
            and supports_arg(pipeline, self.guidance_parameter)
            and values.get("guidance_scale") is not None
        ):
            target[self.guidance_parameter] = values.get("guidance_scale")
        if (
            self.conditioning_scale_parameter
            and supports_arg(pipeline, self.conditioning_scale_parameter)
            and values.get("conditioning_scale") is not None
        ):
            target[self.conditioning_scale_parameter] = values["conditioning_scale"]


IMAGE_PIPELINE_ADAPTERS = {
    "DDPMPipeline": ImagePipelineAdapter(
        "DDPMPipeline",
        frozenset({"unconditional_image"}),
        DDPM_CIFAR10_REPO,
    ),
    "DDIMPipeline": ImagePipelineAdapter(
        "DDIMPipeline",
        frozenset({"unconditional_image"}),
        DDPM_CIFAR10_REPO,
        unconditional_optional_fields=("eta",),
    ),
    "ConsistencyModelPipeline": ImagePipelineAdapter(
        "ConsistencyModelPipeline",
        frozenset({"unconditional_image"}),
        CONSISTENCY_IMAGENET64_REPO,
        unconditional_optional_fields=("class_label",),
    ),
    "QwenImagePipeline": ImagePipelineAdapter(
        "QwenImagePipeline",
        frozenset({"text_to_image"}),
        QWEN_IMAGE_2512_REPO,
        compatible_repos=frozenset({QWEN_IMAGE_2512_PREQUANTIZED_REPO}),
        guidance_parameter="true_cfg_scale",
    ),
    "ZImagePipeline": ImagePipelineAdapter("ZImagePipeline", frozenset({"text_to_image"}), Z_IMAGE_REPO),
    "ZImageImg2ImgPipeline": ImagePipelineAdapter(
        "ZImageImg2ImgPipeline",
        frozenset({"edit_image"}),
        Z_IMAGE_REPO,
        artifact_pipeline_classes=("ZImagePipeline", "ZImageImg2ImgPipeline"),
    ),
    "ZImageInpaintPipeline": ImagePipelineAdapter(
        "ZImageInpaintPipeline",
        frozenset({"inpaint", "outpaint"}),
        Z_IMAGE_REPO,
        artifact_pipeline_classes=("ZImagePipeline", "ZImageInpaintPipeline"),
    ),
    "StableDiffusionXLPipeline": ImagePipelineAdapter(
        "StableDiffusionXLPipeline",
        frozenset({"text_to_image"}),
        SDXL_BASE_REPO,
    ),
    "StableDiffusionXLTurboPipeline": ImagePipelineAdapter(
        "StableDiffusionXLTurboPipeline",
        frozenset({"text_to_image"}),
        SDXL_TURBO_REPO,
        artifact_pipeline_classes=("StableDiffusionXLPipeline",),
        runtime_pipeline_classes=("StableDiffusionXLPipeline",),
        upstream_pipeline_class="StableDiffusionXLPipeline",
        safe_serialization_required=True,
        weight_variant="fp16",
        max_inference_steps=4,
        fixed_guidance_scale=0.0,
    ),
    "StableDiffusionXLInstructPix2PixPipeline": ImagePipelineAdapter(
        "StableDiffusionXLInstructPix2PixPipeline",
        frozenset({"edit_image"}),
        SDXL_INSTRUCT_PIX2PIX_REPO,
        safe_serialization_required=True,
        minimum_image_guidance_scale=1.0,
    ),
    "StableDiffusionXLControlNetPipeline": ImagePipelineAdapter(
        "StableDiffusionXLControlNetPipeline",
        frozenset({"control_image"}),
        SDXL_BASE_REPO,
        artifact_pipeline_classes=("StableDiffusionXLPipeline",),
        safe_serialization_required=True,
        weight_variant="fp16",
        conditioning_kind="controlnet",
        default_conditioning_repo=SDXL_CONTROLNET_CANNY_REPO,
        conditioning_component_class="ControlNetModel",
        conditioning_component_parameter="controlnet",
        conditioning_weight_variant="fp16",
        control_image_parameter="image",
        conditioning_scale_parameter="controlnet_conditioning_scale",
    ),
    "HunyuanDiTControlNetPipeline": ImagePipelineAdapter(
        "HunyuanDiTControlNetPipeline",
        frozenset({"control_image"}),
        HUNYUAN_DIT_DISTILLED_REPO,
        artifact_pipeline_classes=("HunyuanDiTPipeline",),
        safe_serialization_required=True,
        max_inference_steps=50,
        min_output_side=1024,
        max_output_side=1024,
        output_side_step=32,
        max_output_pixels=1024 * 1024,
        max_sequence_length=256,
        conditioning_kind="controlnet",
        default_conditioning_repo=HUNYUAN_DIT_CONTROLNET_CANNY_REPO,
        conditioning_component_class="HunyuanDiT2DControlNetModel",
        conditioning_component_parameter="controlnet",
        conditioning_scale_parameter="controlnet_conditioning_scale",
    ),
    "StableDiffusionXLAdapterPipeline": ImagePipelineAdapter(
        "StableDiffusionXLAdapterPipeline",
        frozenset({"control_image"}),
        SDXL_BASE_REPO,
        artifact_pipeline_classes=("StableDiffusionXLPipeline",),
        safe_serialization_required=True,
        weight_variant="fp16",
        conditioning_kind="t2i_adapter",
        default_conditioning_repo=SDXL_T2I_ADAPTER_CANNY_REPO,
        conditioning_component_class="T2IAdapter",
        conditioning_component_parameter="adapter",
        conditioning_weight_variant="fp16",
        control_image_parameter="image",
        conditioning_scale_parameter="adapter_conditioning_scale",
    ),
    "StableDiffusionXLPAGPipeline": ImagePipelineAdapter(
        "StableDiffusionXLPAGPipeline",
        frozenset({"text_to_image"}),
        SDXL_BASE_REPO,
        artifact_pipeline_classes=("StableDiffusionXLPipeline",),
        safe_serialization_required=True,
        weight_variant="fp16",
    ),
    "StableDiffusionXLPAGImg2ImgPipeline": ImagePipelineAdapter(
        "StableDiffusionXLPAGImg2ImgPipeline",
        frozenset({"edit_image"}),
        SDXL_BASE_REPO,
        artifact_pipeline_classes=("StableDiffusionXLPipeline",),
        safe_serialization_required=True,
        weight_variant="fp16",
    ),
    "StableDiffusionXLPAGInpaintPipeline": ImagePipelineAdapter(
        "StableDiffusionXLPAGInpaintPipeline",
        frozenset({"inpaint"}),
        SDXL_BASE_REPO,
        artifact_pipeline_classes=("StableDiffusionXLPipeline",),
        safe_serialization_required=True,
        weight_variant="fp16",
    ),
    "SanaPipeline": ImagePipelineAdapter(
        "SanaPipeline",
        frozenset({"text_to_image"}),
        SANA_REPO,
        safe_serialization_required=True,
        weight_variant="fp16",
        component_dtype_overrides=(("text_encoder", "bfloat16"), ("vae", "bfloat16")),
        max_sequence_length=300,
    ),
    "SanaSprintPipeline": ImagePipelineAdapter(
        "SanaSprintPipeline",
        frozenset({"text_to_image"}),
        SANA_SPRINT_REPO,
        safe_serialization_required=True,
        max_inference_steps=4,
        max_sequence_length=300,
    ),
    "SanaSprintImg2ImgPipeline": ImagePipelineAdapter(
        "SanaSprintImg2ImgPipeline",
        frozenset({"edit_image"}),
        SANA_SPRINT_REPO,
        artifact_pipeline_classes=("SanaSprintPipeline",),
        safe_serialization_required=True,
        max_inference_steps=4,
        max_sequence_length=300,
    ),
    "PixArtSigmaPipeline": ImagePipelineAdapter(
        "PixArtSigmaPipeline",
        frozenset({"text_to_image"}),
        PIXART_SIGMA_REPO,
        safe_serialization_required=True,
        max_inference_steps=50,
        max_sequence_length=300,
    ),
    "AuraFlowPipeline": ImagePipelineAdapter(
        "AuraFlowPipeline",
        frozenset({"text_to_image"}),
        AURAFLOW_V03_REPO,
        safe_serialization_required=True,
        weight_variant="fp16",
        max_inference_steps=50,
        max_output_side=1536,
        max_sequence_length=256,
    ),
    "ChromaPipeline": ImagePipelineAdapter(
        "ChromaPipeline",
        frozenset({"text_to_image"}),
        CHROMA1_HD_REPO,
        safe_serialization_required=True,
        max_inference_steps=40,
        max_output_side=1024,
        max_sequence_length=512,
    ),
    "CogView3PlusPipeline": ImagePipelineAdapter(
        "CogView3PlusPipeline",
        frozenset({"text_to_image"}),
        COGVIEW3_PLUS_REPO,
        safe_serialization_required=True,
        max_inference_steps=50,
        min_output_side=512,
        max_output_side=2048,
        output_side_step=32,
        max_sequence_length=224,
    ),
    "CogView4Pipeline": ImagePipelineAdapter(
        "CogView4Pipeline",
        frozenset({"text_to_image"}),
        COGVIEW4_6B_REPO,
        safe_serialization_required=True,
        max_inference_steps=50,
        min_output_side=512,
        max_output_side=2048,
        output_side_step=32,
        max_output_pixels=2**21,
        max_sequence_length=1024,
    ),
    "ErnieImagePipeline": ImagePipelineAdapter(
        "ErnieImagePipeline",
        frozenset({"text_to_image"}),
        ERNIE_IMAGE_TURBO_REPO,
        safe_serialization_required=True,
        max_inference_steps=8,
        min_output_side=1024,
        max_output_side=1024,
        output_side_step=32,
        max_output_pixels=1024 * 1024,
        fixed_guidance_scale=1.0,
        max_sequence_length=2048,
    ),
    "DreamLitePipeline": ImagePipelineAdapter(
        "DreamLitePipeline",
        frozenset({"text_to_image", "edit_image"}),
        DREAMLITE_BASE_REPO,
        safe_serialization_required=True,
        max_inference_steps=50,
        minimum_image_guidance_scale=0.0,
        max_sequence_length=200,
    ),
    "DreamLiteMobilePipeline": ImagePipelineAdapter(
        "DreamLiteMobilePipeline",
        frozenset({"text_to_image", "edit_image"}),
        DREAMLITE_MOBILE_REPO,
        safe_serialization_required=True,
        max_inference_steps=8,
        fixed_guidance_scale=0.0,
        guidance_parameter=None,
        ignored_generation_parameters=frozenset({"image_guidance_scale"}),
        max_sequence_length=200,
    ),
    "StableDiffusionXLImg2ImgPipeline": ImagePipelineAdapter(
        "StableDiffusionXLImg2ImgPipeline",
        frozenset({"edit_image"}),
        SDXL_BASE_REPO,
        artifact_pipeline_classes=("StableDiffusionXLPipeline", "StableDiffusionXLImg2ImgPipeline"),
    ),
    "StableDiffusionXLInpaintPipeline": ImagePipelineAdapter(
        "StableDiffusionXLInpaintPipeline",
        frozenset({"inpaint", "outpaint"}),
        SDXL_BASE_REPO,
        artifact_pipeline_classes=("StableDiffusionXLPipeline", "StableDiffusionXLInpaintPipeline"),
    ),
    "StableDiffusionPipeline": ImagePipelineAdapter(
        "StableDiffusionPipeline",
        frozenset({"text_to_image"}),
        SD15_BASE_REPO,
    ),
    "StableDiffusionControlNetPipeline": ImagePipelineAdapter(
        "StableDiffusionControlNetPipeline",
        frozenset({"control_image"}),
        SD15_BASE_REPO,
        artifact_pipeline_classes=("StableDiffusionPipeline",),
        conditioning_kind="controlnet",
        default_conditioning_repo=SD15_CONTROLNET_CANNY_REPO,
        conditioning_component_class="ControlNetModel",
        conditioning_component_parameter="controlnet",
        control_image_parameter="image",
        conditioning_scale_parameter="controlnet_conditioning_scale",
    ),
    "StableDiffusionImg2ImgPipeline": ImagePipelineAdapter(
        "StableDiffusionImg2ImgPipeline",
        frozenset({"edit_image"}),
        SD15_BASE_REPO,
        artifact_pipeline_classes=("StableDiffusionPipeline", "StableDiffusionImg2ImgPipeline"),
    ),
    "StableDiffusionInpaintPipeline": ImagePipelineAdapter(
        "StableDiffusionInpaintPipeline",
        frozenset({"inpaint", "outpaint"}),
        SD15_BASE_REPO,
        artifact_pipeline_classes=("StableDiffusionPipeline", "StableDiffusionInpaintPipeline"),
    ),
    "LatentConsistencyModelPipeline": ImagePipelineAdapter(
        "LatentConsistencyModelPipeline",
        frozenset({"text_to_image"}),
        LCM_DREAMSHAPER_REPO,
    ),
    "StableDiffusionPAGPipeline": ImagePipelineAdapter(
        "StableDiffusionPAGPipeline",
        frozenset({"text_to_image"}),
        SD15_BASE_REPO,
        artifact_pipeline_classes=("StableDiffusionPipeline", "StableDiffusionPAGPipeline"),
    ),
    "MarigoldDepthPipeline": ImagePipelineAdapter(
        "MarigoldDepthPipeline",
        frozenset({"depth_estimation"}),
        MARIGOLD_DEPTH_LCM_REPO,
    ),
    "FluxPipeline": ImagePipelineAdapter(
        "FluxPipeline",
        frozenset({"text_to_image"}),
        FLUX_SCHNELL_REPO,
        compatible_repos=frozenset({FLUX_DEV_REPO, FLUX_DEV_FP8_REPO, FLUX_KREA_REPO}),
        guidance_parameter="true_cfg_scale",
    ),
    "Flux2KleinPipeline": ImagePipelineAdapter(
        "Flux2KleinPipeline",
        frozenset({"text_to_image", "edit_image", "multi_image_reference_edit"}),
        FLUX2_KLEIN_REPO,
        max_reference_images=8,
    ),
    "Flux2KleinInpaintPipeline": ImagePipelineAdapter(
        "Flux2KleinInpaintPipeline",
        frozenset({"inpaint", "outpaint"}),
        FLUX2_KLEIN_REPO,
        artifact_pipeline_classes=("Flux2KleinPipeline", "Flux2KleinInpaintPipeline"),
    ),
    "FluxImg2ImgPipeline": ImagePipelineAdapter(
        "FluxImg2ImgPipeline",
        frozenset({"edit_image"}),
        FLUX_DEV_REPO,
        compatible_repos=frozenset({FLUX_DEV_FP8_REPO}),
        artifact_pipeline_classes=("FluxPipeline", "FluxImg2ImgPipeline"),
        guidance_parameter="true_cfg_scale",
    ),
    "FluxInpaintPipeline": ImagePipelineAdapter(
        "FluxInpaintPipeline",
        frozenset({"inpaint"}),
        FLUX_DEV_REPO,
        compatible_repos=frozenset({FLUX_DEV_FP8_REPO}),
        artifact_pipeline_classes=("FluxPipeline", "FluxInpaintPipeline"),
        guidance_parameter="true_cfg_scale",
    ),
    "FluxFillPipeline": ImagePipelineAdapter("FluxFillPipeline", frozenset({"inpaint", "outpaint"}), FLUX_FILL_REPO),
    "FluxControlPipeline": ImagePipelineAdapter(
        "FluxControlPipeline",
        frozenset({"control_image"}),
        FLUX_DEPTH_REPO,
        compatible_repos=frozenset({FLUX_CANNY_REPO, FLUX_CANNY_REPAIR_REPO}),
    ),
    "FluxKontextPipeline": ImagePipelineAdapter(
        "FluxKontextPipeline",
        frozenset({"edit_image", "multi_image_reference_edit"}),
        FLUX_KONTEXT_REPO,
        compatible_repos=frozenset({FLUX_KONTEXT_NVFP4_REPO}),
        guidance_parameter="true_cfg_scale",
        multi_image_strategy="stitch_horizontal",
        max_reference_images=8,
    ),
    "FluxKontextInpaintPipeline": ImagePipelineAdapter(
        "FluxKontextInpaintPipeline",
        frozenset({"inpaint", "outpaint"}),
        FLUX_KONTEXT_REPO,
        compatible_repos=frozenset({FLUX_KONTEXT_NVFP4_REPO}),
        artifact_pipeline_classes=("FluxKontextPipeline", "FluxKontextInpaintPipeline"),
        guidance_parameter="true_cfg_scale",
    ),
    # Virtual adapter class: FLUX Redux is a prior that supplies embeddings to
    # a base FLUX pipeline, not a standalone img2img checkpoint.
    "FluxReduxPipeline": ImagePipelineAdapter(
        "FluxReduxPipeline",
        frozenset({"edit_image", "multi_image_reference_edit"}),
        FLUX_REDUX_REPO,
        artifact_pipeline_classes=("FluxPriorReduxPipeline",),
        runtime_pipeline_classes=("FluxReduxPipelineBundle",),
        # Current Diffusers performs the documented per-reference scaling and
        # weighted sum inside FluxPriorReduxPipeline. Keep the references as a
        # list and delegate the conditioning math to the upstream pipeline.
        multi_image_strategy="upstream_weighted_sum",
        max_reference_images=8,
    ),
    "QwenImageEditInpaintPipeline": ImagePipelineAdapter(
        "QwenImageEditInpaintPipeline",
        frozenset({"inpaint", "outpaint"}),
        QWEN_IMAGE_EDIT_REPO,
        compatible_repos=frozenset({QWEN_IMAGE_EDIT_PREQUANTIZED_REPO}),
        artifact_pipeline_classes=("QwenImageEditPipeline", "QwenImageEditInpaintPipeline"),
        guidance_parameter="true_cfg_scale",
    ),
    "QwenImageImg2ImgPipeline": ImagePipelineAdapter(
        "QwenImageImg2ImgPipeline",
        frozenset({"edit_image"}),
        QWEN_IMAGE_2512_REPO,
        compatible_repos=frozenset({QWEN_IMAGE_2512_PREQUANTIZED_REPO}),
        artifact_pipeline_classes=("QwenImagePipeline", "QwenImageImg2ImgPipeline"),
        guidance_parameter="true_cfg_scale",
    ),
    "QwenImageInpaintPipeline": ImagePipelineAdapter(
        "QwenImageInpaintPipeline",
        frozenset({"inpaint", "outpaint"}),
        QWEN_IMAGE_2512_REPO,
        compatible_repos=frozenset({QWEN_IMAGE_2512_PREQUANTIZED_REPO}),
        artifact_pipeline_classes=("QwenImagePipeline", "QwenImageInpaintPipeline"),
        guidance_parameter="true_cfg_scale",
    ),
    "QwenImageEditPipeline": ImagePipelineAdapter(
        "QwenImageEditPipeline",
        frozenset({"edit_image"}),
        QWEN_IMAGE_EDIT_REPO,
        compatible_repos=frozenset({QWEN_IMAGE_EDIT_PREQUANTIZED_REPO}),
        guidance_parameter="true_cfg_scale",
    ),
    "QwenImageEditPlusPipeline": ImagePipelineAdapter(
        "QwenImageEditPlusPipeline",
        frozenset({"edit_image", "multi_image_reference_edit"}),
        QWEN_IMAGE_EDIT_PLUS_REPO,
        guidance_parameter="true_cfg_scale",
        max_reference_images=8,
    ),
}
IMAGE_PIPELINE_CLASSES = list(IMAGE_PIPELINE_ADAPTERS)
IMAGE_PIPELINE_MODES = {name: set(adapter.modes) for name, adapter in IMAGE_PIPELINE_ADAPTERS.items()}
IMAGE_PIPELINE_MODE_OPTIONS = [
    "unconditional_image",
    "depth_estimation",
    "text_to_image",
    "edit_image",
    "multi_image_reference_edit",
    "inpaint",
    "outpaint",
    "control_image",
]
DIFFUSERS_IMAGE_OFFLOAD_MODES = [
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
]
QUANT_COMPONENTS = ["transformer", "transformer_2", "text_encoder", "text_encoder_2", "vae"]

IMAGE_ACTION_MODES = {
    "UnconditionalGenerate": ("unconditional_image",),
    "PredictMap": ("depth_estimation",),
    "Generate": ("text_to_image",),
    "Edit": ("edit_image", "multi_image_reference_edit"),
    "Inpaint": ("inpaint", "outpaint"),
    "ControlGenerate": ("control_image",),
}

_IMAGE_CONTRACT_VISIBILITY_FIELDS = (
    "negative_prompt",
    "width",
    "height",
    "guidance_scale",
    "strength",
    "padding_mask_crop",
    "max_sequence_length",
    "reference_strength",
    "image_guidance_scale",
    "pag_scale",
    "pag_adaptive_scale",
    "conditioning_scale",
)


@dataclass(frozen=True)
class ImageModeFieldContract:
    visible_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.visible_fields)) != len(self.visible_fields) or any(
            field not in _IMAGE_CONTRACT_VISIBILITY_FIELDS for field in self.visible_fields
        ):
            raise ValueError("Image mode contracts must declare unique reviewed visibility fields.")

    def field_param_overlay(self) -> dict[str, dict[str, bool]]:
        return {field: {"hidden": field not in self.visible_fields} for field in _IMAGE_CONTRACT_VISIBILITY_FIELDS}


def _image_field_contract(*visible_fields: str) -> ImageModeFieldContract:
    return ImageModeFieldContract(visible_fields=visible_fields)


_NEGATIVE_SIZE_GUIDANCE_SEQUENCE = (
    "negative_prompt",
    "width",
    "height",
    "guidance_scale",
    "max_sequence_length",
)
_SIZE_GUIDANCE_SEQUENCE = ("width", "height", "guidance_scale", "max_sequence_length")
_NEGATIVE_SIZE_GUIDANCE_STRENGTH_SEQUENCE = (
    "negative_prompt",
    "width",
    "height",
    "guidance_scale",
    "strength",
    "max_sequence_length",
)
_SIZE_GUIDANCE_STRENGTH_SEQUENCE = (
    "width",
    "height",
    "guidance_scale",
    "strength",
    "max_sequence_length",
)
_NEGATIVE_SIZE_GUIDANCE_STRENGTH_CROP_SEQUENCE = (
    "negative_prompt",
    "width",
    "height",
    "guidance_scale",
    "strength",
    "padding_mask_crop",
    "max_sequence_length",
)
_SIZE_GUIDANCE_STRENGTH_CROP_SEQUENCE = (
    "width",
    "height",
    "guidance_scale",
    "strength",
    "padding_mask_crop",
    "max_sequence_length",
)

IMAGE_MODE_FIELD_CONTRACTS = {
    "DDPMPipeline": {
        "unconditional_image": _image_field_contract(),
    },
    "DDIMPipeline": {
        "unconditional_image": _image_field_contract(),
    },
    "ConsistencyModelPipeline": {
        "unconditional_image": _image_field_contract(),
    },
    "QwenImagePipeline": {
        "text_to_image": _image_field_contract(*_NEGATIVE_SIZE_GUIDANCE_SEQUENCE),
    },
    "ZImagePipeline": {
        "text_to_image": _image_field_contract(*_NEGATIVE_SIZE_GUIDANCE_SEQUENCE),
    },
    "ZImageImg2ImgPipeline": {
        "edit_image": _image_field_contract(*_NEGATIVE_SIZE_GUIDANCE_STRENGTH_SEQUENCE),
    },
    "ZImageInpaintPipeline": {
        mode: _image_field_contract(*_NEGATIVE_SIZE_GUIDANCE_STRENGTH_SEQUENCE) for mode in ("inpaint", "outpaint")
    },
    "StableDiffusionXLPipeline": {
        "text_to_image": _image_field_contract("negative_prompt", "width", "height", "guidance_scale"),
    },
    "StableDiffusionXLTurboPipeline": {
        "text_to_image": _image_field_contract("width", "height"),
    },
    "StableDiffusionXLInstructPix2PixPipeline": {
        "edit_image": _image_field_contract(
            "negative_prompt", "width", "height", "guidance_scale", "image_guidance_scale"
        ),
    },
    "StableDiffusionXLControlNetPipeline": {
        "control_image": _image_field_contract(
            "negative_prompt", "width", "height", "guidance_scale", "conditioning_scale"
        ),
    },
    "HunyuanDiTControlNetPipeline": {
        "control_image": _image_field_contract(
            "negative_prompt", "width", "height", "guidance_scale", "conditioning_scale"
        ),
    },
    "StableDiffusionXLAdapterPipeline": {
        "control_image": _image_field_contract(
            "negative_prompt", "width", "height", "guidance_scale", "conditioning_scale"
        ),
    },
    "StableDiffusionXLPAGPipeline": {
        "text_to_image": _image_field_contract(
            "negative_prompt", "width", "height", "guidance_scale", "pag_scale", "pag_adaptive_scale"
        ),
    },
    "StableDiffusionXLPAGImg2ImgPipeline": {
        "edit_image": _image_field_contract(
            "negative_prompt", "guidance_scale", "strength", "pag_scale", "pag_adaptive_scale"
        ),
    },
    "StableDiffusionXLPAGInpaintPipeline": {
        "inpaint": _image_field_contract(
            "negative_prompt",
            "width",
            "height",
            "guidance_scale",
            "strength",
            "padding_mask_crop",
            "pag_scale",
            "pag_adaptive_scale",
        ),
    },
    "SanaPipeline": {
        "text_to_image": _image_field_contract(*_NEGATIVE_SIZE_GUIDANCE_SEQUENCE),
    },
    "SanaSprintPipeline": {
        "text_to_image": _image_field_contract("width", "height", "guidance_scale", "max_sequence_length"),
    },
    "SanaSprintImg2ImgPipeline": {
        "edit_image": _image_field_contract("width", "height", "guidance_scale", "strength", "max_sequence_length"),
    },
    "PixArtSigmaPipeline": {
        "text_to_image": _image_field_contract(*_NEGATIVE_SIZE_GUIDANCE_SEQUENCE),
    },
    "AuraFlowPipeline": {
        "text_to_image": _image_field_contract(*_NEGATIVE_SIZE_GUIDANCE_SEQUENCE),
    },
    "ChromaPipeline": {
        "text_to_image": _image_field_contract(*_NEGATIVE_SIZE_GUIDANCE_SEQUENCE),
    },
    "CogView3PlusPipeline": {
        "text_to_image": _image_field_contract(*_NEGATIVE_SIZE_GUIDANCE_SEQUENCE),
    },
    "CogView4Pipeline": {
        "text_to_image": _image_field_contract(*_NEGATIVE_SIZE_GUIDANCE_SEQUENCE),
    },
    "ErnieImagePipeline": {
        "text_to_image": _image_field_contract("width", "height"),
    },
    "DreamLitePipeline": {
        "text_to_image": _image_field_contract(
            "negative_prompt", "width", "height", "guidance_scale", "max_sequence_length"
        ),
        "edit_image": _image_field_contract(
            "negative_prompt", "width", "height", "guidance_scale", "image_guidance_scale"
        ),
    },
    "DreamLiteMobilePipeline": {
        "text_to_image": _image_field_contract("width", "height", "max_sequence_length"),
        "edit_image": _image_field_contract("width", "height"),
    },
    "StableDiffusionXLImg2ImgPipeline": {
        "edit_image": _image_field_contract("negative_prompt", "guidance_scale", "strength"),
    },
    "StableDiffusionXLInpaintPipeline": {
        mode: _image_field_contract(
            "negative_prompt", "width", "height", "guidance_scale", "strength", "padding_mask_crop"
        )
        for mode in ("inpaint", "outpaint")
    },
    "StableDiffusionPipeline": {
        "text_to_image": _image_field_contract("negative_prompt", "width", "height", "guidance_scale"),
    },
    "StableDiffusionControlNetPipeline": {
        "control_image": _image_field_contract(
            "negative_prompt", "width", "height", "guidance_scale", "conditioning_scale"
        ),
    },
    "StableDiffusionImg2ImgPipeline": {
        "edit_image": _image_field_contract("negative_prompt", "guidance_scale", "strength"),
    },
    "StableDiffusionInpaintPipeline": {
        mode: _image_field_contract(
            "negative_prompt", "width", "height", "guidance_scale", "strength", "padding_mask_crop"
        )
        for mode in ("inpaint", "outpaint")
    },
    "LatentConsistencyModelPipeline": {
        "text_to_image": _image_field_contract("width", "height", "guidance_scale"),
    },
    "StableDiffusionPAGPipeline": {
        "text_to_image": _image_field_contract(
            "negative_prompt", "width", "height", "guidance_scale", "pag_scale", "pag_adaptive_scale"
        ),
    },
    "MarigoldDepthPipeline": {
        "depth_estimation": _image_field_contract(),
    },
    "FluxPipeline": {
        "text_to_image": _image_field_contract(*_NEGATIVE_SIZE_GUIDANCE_SEQUENCE),
    },
    "Flux2KleinPipeline": {
        mode: _image_field_contract(*_SIZE_GUIDANCE_SEQUENCE)
        for mode in ("text_to_image", "edit_image", "multi_image_reference_edit")
    },
    "Flux2KleinInpaintPipeline": {
        mode: _image_field_contract(*_SIZE_GUIDANCE_STRENGTH_CROP_SEQUENCE) for mode in ("inpaint", "outpaint")
    },
    "FluxImg2ImgPipeline": {
        "edit_image": _image_field_contract(*_NEGATIVE_SIZE_GUIDANCE_STRENGTH_SEQUENCE),
    },
    "FluxInpaintPipeline": {
        "inpaint": _image_field_contract(*_NEGATIVE_SIZE_GUIDANCE_STRENGTH_CROP_SEQUENCE),
    },
    "FluxFillPipeline": {
        mode: _image_field_contract(*_SIZE_GUIDANCE_STRENGTH_SEQUENCE) for mode in ("inpaint", "outpaint")
    },
    "FluxControlPipeline": {
        "control_image": _image_field_contract(*_SIZE_GUIDANCE_SEQUENCE),
    },
    "FluxKontextPipeline": {
        mode: _image_field_contract(*_NEGATIVE_SIZE_GUIDANCE_SEQUENCE)
        for mode in ("edit_image", "multi_image_reference_edit")
    },
    "FluxKontextInpaintPipeline": {
        mode: _image_field_contract(*_NEGATIVE_SIZE_GUIDANCE_STRENGTH_CROP_SEQUENCE)
        for mode in ("inpaint", "outpaint")
    },
    "FluxReduxPipeline": {
        "edit_image": _image_field_contract(*_NEGATIVE_SIZE_GUIDANCE_SEQUENCE),
        "multi_image_reference_edit": _image_field_contract(*_NEGATIVE_SIZE_GUIDANCE_SEQUENCE, "reference_strength"),
    },
    "QwenImageEditInpaintPipeline": {
        mode: _image_field_contract(*_NEGATIVE_SIZE_GUIDANCE_STRENGTH_CROP_SEQUENCE)
        for mode in ("inpaint", "outpaint")
    },
    "QwenImageImg2ImgPipeline": {
        "edit_image": _image_field_contract(*_NEGATIVE_SIZE_GUIDANCE_STRENGTH_SEQUENCE),
    },
    "QwenImageInpaintPipeline": {
        mode: _image_field_contract(*_NEGATIVE_SIZE_GUIDANCE_STRENGTH_CROP_SEQUENCE)
        for mode in ("inpaint", "outpaint")
    },
    "QwenImageEditPipeline": {
        "edit_image": _image_field_contract(*_NEGATIVE_SIZE_GUIDANCE_SEQUENCE),
    },
    "QwenImageEditPlusPipeline": {
        mode: _image_field_contract(*_NEGATIVE_SIZE_GUIDANCE_SEQUENCE)
        for mode in ("edit_image", "multi_image_reference_edit")
    },
}


def get_image_mode_field_contract(adapter: ImagePipelineAdapter, mode: str) -> ImageModeFieldContract:
    contracts = IMAGE_MODE_FIELD_CONTRACTS.get(adapter.pipeline_class)
    if contracts is None or tuple(contracts) != adapter.mode_options:
        raise RuntimeError(f"Image adapter {adapter.pipeline_class} has an incomplete reviewed field contract.")
    contract = contracts.get(mode)
    if contract is None:
        raise ValueError(f"{adapter.pipeline_class} does not support image mode {mode}.")
    return contract


_REMOVED_IMAGE_PIPELINE_ERRORS = {
    "FluxControlNetPipeline": (
        "FluxControlNetPipeline requires a separately loaded FluxControlNetModel, but the generic Diffusers image "
        "loader does not yet expose that component-assembly contract. Use FluxControlPipeline for the self-contained "
        "FLUX Depth/Canny checkpoints until generic ControlNet assembly is available."
    ),
    "StableDiffusionAdapterPipeline": (
        "StableDiffusionAdapterPipeline requires a separately loaded T2IAdapter. The reviewed official SD1.5 "
        "Canny adapter currently publishes legacy PyTorch .bin weights only, so MoDiff cannot expose an exact "
        "runnable pair under its safetensors-only auxiliary policy."
    ),
}


def repo_value(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("value") or "")
    return str(value or "")


def get_image_pipeline_adapter(name: Any) -> ImagePipelineAdapter:
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError("A registered Diffusers image pipeline class is required.")
    removed_reason = _REMOVED_IMAGE_PIPELINE_ERRORS.get(name)
    if removed_reason:
        raise ValueError(removed_reason)
    adapter = IMAGE_PIPELINE_ADAPTERS.get(name)
    if adapter is None:
        supported = ", ".join(IMAGE_PIPELINE_CLASSES)
        raise ValueError(f"Unsupported Diffusers image pipeline class {name!r}. Supported classes: {supported}.")
    return adapter


def _loader_image_pipeline_adapter(values: Any) -> ImagePipelineAdapter:
    if not isinstance(values, dict):
        raise ValueError("Diffusers image loader values must be an object.")
    return get_image_pipeline_adapter(values.get("pipeline_class"))


def _loader_image_mode(values: Any, adapter: ImagePipelineAdapter) -> str:
    mode = values.get("mode") if isinstance(values, dict) else None
    if not isinstance(mode, str) or not mode or mode != mode.strip():
        raise ValueError("A registered Diffusers image mode is required.")
    if mode not in adapter.modes:
        supported = ", ".join(adapter.mode_options)
        raise ValueError(f"{adapter.pipeline_class} does not support {mode}. Supported modes: {supported}.")
    return mode


def _canonical_image_model_source(source: Any, *, label: str = "Diffusers image model") -> str:
    if not isinstance(source, str) or not source or source != source.strip():
        raise ValueError(f"{label} source must be exactly hub or local.")
    normalized = source.casefold()
    if normalized not in _IMAGE_MODEL_SOURCES:
        raise ValueError(f"{label} source must be exactly hub or local.")
    return normalized


def _managed_image_repositories() -> dict[str, str]:
    return {
        repository.casefold(): repository
        for registered_adapter in IMAGE_PIPELINE_ADAPTERS.values()
        for repository in registered_adapter.managed_repos
    }


def _validated_image_hub_repository(value: str, *, label: str) -> str:
    if value.count("/") != 1:
        raise ValueError(f"{label} must use an exact Hugging Face namespace/repository ID.")
    try:
        validate_hf_repo_id(value)
    except ValueError as error:
        raise ValueError(f"{label} must use an exact Hugging Face namespace/repository ID.") from error
    try:
        resolves_locally = Path(value).expanduser().exists()
    except (OSError, RuntimeError) as error:
        raise ValueError(f"{label} could not be validated as a Hugging Face repository ID.") from error
    if resolves_locally:
        raise ValueError(
            f"{label} resolves to an existing local filesystem target. Select source=local for local models."
        )
    return value


def resolve_image_model_selection(adapter: ImagePipelineAdapter, value: Any):
    """Return one canonical Hub/local selection for the chosen adapter."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return {"source": "hub", "value": adapter.default_repo}
    if isinstance(value, dict):
        source = _canonical_image_model_source(value.get("source"))
        raw_selected = value.get("value")
        if not isinstance(raw_selected, str):
            raise ValueError("Diffusers image model value must be a repository ID or local path string.")
        selected = raw_selected.strip()
        if not selected:
            if source == "hub":
                return {"source": "hub", "value": adapter.default_repo}
            raise ValueError("A local Diffusers image model path is required.")
    elif isinstance(value, str):
        source = "hub"
        selected = value.strip()
    else:
        raise ValueError("Diffusers image model selection must be a repository ID or a hub/local selection object.")

    if source == "local":
        return {"source": "local", "value": selected}

    selected = _validated_image_hub_repository(selected, label="Diffusers image model repository")
    managed_repos = _managed_image_repositories()
    selected_key = selected.casefold()
    compatible_repos = {repo.casefold() for repo in adapter.managed_repos}
    if selected_key in managed_repos and selected_key not in compatible_repos:
        return {"source": "hub", "value": adapter.default_repo}
    return {"source": "hub", "value": managed_repos.get(selected_key, selected)}


def _normalize_image_revision(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or value != value.strip():
        raise ValueError("Diffusers image revision must be an exact trimmed string.")
    return value


def resolve_image_pipeline_revision(model_selection: Any, revision: Any) -> str:
    """Resolve one immutable Hub revision or reject unsupported local execution."""

    if not isinstance(model_selection, dict):
        raise ValueError("Diffusers image model selection must be normalized before revision resolution.")
    source = _canonical_image_model_source(model_selection.get("source"))
    model_id = model_selection.get("value")
    if not isinstance(model_id, str) or not model_id or model_id != model_id.strip():
        raise ValueError("Diffusers image model value must be a nonblank canonical string.")
    requested = _normalize_image_revision(revision)
    if source == "local":
        raise ValueError(
            "Local standard Diffusers pipeline loading is contract-only until MoDiff has a reviewed local "
            "pipeline-directory index. Install or select a pinned Hub pipeline through Model Manager."
        )

    pin = catalog_repository_pin(model_id)
    if pin is not None:
        expected = catalog_revision(model_id)
        if requested and requested != expected:
            raise ValueError(
                f"Cataloged Diffusers image repository {model_id!r} must use its reviewed commit {expected}; "
                f"received {requested!r}."
            )
        return str(expected)
    if not requested:
        raise ValueError(
            f"Custom Diffusers image repository {model_id!r} requires an explicit lowercase 40-character commit."
        )
    if requested != requested.lower() or not IMMUTABLE_HUB_REVISION.fullmatch(requested):
        raise ValueError("A custom Diffusers image repository revision must be a lowercase 40-character commit.")
    return requested


def resolve_image_conditioning_selection(
    adapter: ImagePipelineAdapter,
    kind: Any,
    model_selection: Any,
    revision: Any,
) -> tuple[dict[str, str] | None, str | None]:
    """Resolve one immutable, safetensors-only auxiliary image component."""

    if adapter.conditioning_kind is None:
        if kind not in (None, "", "none"):
            raise ValueError(f"{adapter.pipeline_class} does not accept an auxiliary conditioning component.")
        if revision not in (None, ""):
            raise ValueError(f"{adapter.pipeline_class} does not accept an auxiliary conditioning revision.")
        return None, None

    if kind in (None, ""):
        kind = adapter.conditioning_kind
    if kind != adapter.conditioning_kind:
        raise ValueError(
            f"{adapter.pipeline_class} requires conditioning_kind={adapter.conditioning_kind!r}."
        )
    if model_selection in (None, "", {}):
        model_selection = {"source": "hub", "value": adapter.default_conditioning_repo}
    if isinstance(model_selection, str):
        source = "hub"
        selected = model_selection
    elif isinstance(model_selection, dict):
        source = _canonical_image_model_source(
            model_selection.get("source"), label="Diffusers image conditioning model"
        )
        selected = model_selection.get("value")
    else:
        raise ValueError("Diffusers image conditioning model must be a Hub selection object.")
    if source != "hub":
        raise ValueError(
            "Local image conditioning components remain contract-only until a reviewed local safetensors index exists."
        )
    if not isinstance(selected, str) or not selected or selected != selected.strip():
        raise ValueError("Diffusers image conditioning model requires an exact repository ID.")
    repository = _validated_image_hub_repository(
        selected, label="Diffusers image conditioning model repository"
    )
    pin = catalog_repository_pin(repository)
    canonical_repository = str(pin.get("repo")) if pin is not None else repository
    requested_revision = _normalize_image_revision(revision)
    if pin is not None:
        expected_revision = catalog_revision(canonical_repository)
        if requested_revision and requested_revision != expected_revision:
            raise ValueError(
                f"Cataloged conditioning repository {canonical_repository!r} must use its reviewed commit "
                f"{expected_revision}; received {requested_revision!r}."
            )
        resolved_revision = str(expected_revision)
    else:
        if not requested_revision:
            raise ValueError(
                f"Custom conditioning repository {repository!r} requires an explicit lowercase 40-character commit."
            )
        if requested_revision != requested_revision.lower() or not IMMUTABLE_HUB_REVISION.fullmatch(
            requested_revision
        ):
            raise ValueError("A custom conditioning revision must be a lowercase 40-character commit.")
        resolved_revision = requested_revision
    return {"source": "hub", "value": canonical_repository}, resolved_revision


def image_model_field_options(adapter: ImagePipelineAdapter) -> dict[str, Any]:
    classes = list(adapter.model_filter_classes)
    return {
        "noValidation": True,
        "sources": ["hub", "local"],
        "filter": {
            "hub": {"className": classes},
            "local": {"className": classes},
        },
    }


def image_pipeline_contract(adapter: ImagePipelineAdapter, mode: str) -> dict[str, Any]:
    field_contract = get_image_mode_field_contract(adapter, mode)
    field_params = field_contract.field_param_overlay()
    if (
        adapter.min_output_side != 16
        or adapter.max_output_side != 2048
        or adapter.output_side_step != 16
    ):
        for field in ("width", "height"):
            field_params[field] = {
                **field_params[field],
                "min": adapter.min_output_side,
                "max": adapter.max_output_side,
                "step": adapter.output_side_step,
            }
    if adapter.max_sequence_length != 512:
        field_params["max_sequence_length"] = {
            **field_params["max_sequence_length"],
            "max": adapter.max_sequence_length,
        }
    actions = {
        action: [candidate for candidate in adapter.mode_options if candidate in accepted_modes]
        for action, accepted_modes in IMAGE_ACTION_MODES.items()
    }
    contract = {
        "schemaVersion": 1,
        "library": "diffusers",
        "mediaKind": "image",
        "pipelineClass": adapter.pipeline_class,
        "mode": mode,
        "modes": list(adapter.mode_options),
        "actions": {action: modes for action, modes in actions.items() if modes},
        "fieldParams": field_params,
    }
    if adapter.max_output_pixels != _MAX_IMAGE_OUTPUT_PIXELS:
        contract["maxOutputPixels"] = adapter.max_output_pixels
    if mode == "unconditional_image":
        optional_fields = set(adapter.unconditional_optional_fields)
        contract["actionFieldParams"] = {
            "eta": {"hidden": "eta" not in optional_fields},
            "class_label": {"hidden": "class_label" not in optional_fields},
        }
    if mode == "depth_estimation":
        contract["predictionMap"] = {
            "schemaVersion": 1,
            "kinds": ["depth"],
            "semantics": "relative_depth",
            "layout": "NHWC",
            "dtype": "float32",
            "valueRange": [0.0, 1.0],
            "nearValue": 0.0,
            "farValue": 1.0,
        }
    return contract


DEFAULT_IMAGE_PIPELINE_CONTRACT = image_pipeline_contract(
    IMAGE_PIPELINE_ADAPTERS["FluxPipeline"],
    "text_to_image",
)
DEFAULT_UNCONDITIONAL_IMAGE_PIPELINE_CONTRACT = image_pipeline_contract(
    IMAGE_PIPELINE_ADAPTERS["DDPMPipeline"],
    "unconditional_image",
)
DEFAULT_PREDICTION_MAP_PIPELINE_CONTRACT = image_pipeline_contract(
    IMAGE_PIPELINE_ADAPTERS["MarigoldDepthPipeline"],
    "depth_estimation",
)


def _tag_image_pipeline(
    pipeline: Any,
    adapter: ImagePipelineAdapter,
    mode: str,
    repo: str,
    source: str,
    revision: str | None,
    *,
    conditioning_repo: str | None = None,
    conditioning_revision: str | None = None,
) -> None:
    setattr(pipeline, "_modiff_image_adapter", adapter)
    setattr(pipeline, "_modiff_image_pipeline_class", adapter.pipeline_class)
    setattr(pipeline, "_modiff_image_mode", mode)
    setattr(pipeline, "_modiff_image_repo", repo)
    setattr(pipeline, "_modiff_image_source", source)
    setattr(pipeline, "_modiff_image_revision", revision)
    if adapter.conditioning_kind is not None:
        setattr(pipeline, "_modiff_conditioning_kind", adapter.conditioning_kind)
        setattr(pipeline, "_modiff_conditioning_component_class", adapter.conditioning_component_class)
        setattr(pipeline, "_modiff_conditioning_repo", conditioning_repo)
        setattr(pipeline, "_modiff_conditioning_revision", conditioning_revision)


def _image_pipeline_adapter(pipeline: Any) -> ImagePipelineAdapter:
    tag_names = (
        "_modiff_image_pipeline_class",
        "_modiff_image_mode",
        "_modiff_image_repo",
        "_modiff_image_source",
        "_modiff_image_revision",
    )
    has_any_tag = any(hasattr(pipeline, name) for name in (*tag_names, "_modiff_image_adapter"))
    has_all_tags = all(hasattr(pipeline, name) for name in tag_names)
    runtime_name = type(pipeline).__name__
    adapter_hint = getattr(pipeline, "_modiff_image_adapter", None)
    if has_any_tag:
        if not has_all_tags:
            missing = ", ".join(
                name.removeprefix("_modiff_image_") for name in tag_names if not hasattr(pipeline, name)
            )
            raise ValueError(
                f"Diffusers image pipeline identity is incomplete; missing canonical tags: {missing}. "
                "Reconnect it through Load Diffusers Image Pipeline."
            )
        adapter = get_image_pipeline_adapter(getattr(pipeline, "_modiff_image_pipeline_class"))
        hinted_name = getattr(adapter_hint, "pipeline_class", None)
        if adapter_hint is not None and hinted_name != adapter.pipeline_class:
            raise ValueError(
                "Diffusers image pipeline identity is inconsistent: adapter hint and pipeline-class tag disagree."
            )
        if runtime_name not in adapter.allowed_runtime_classes:
            allowed = ", ".join(adapter.allowed_runtime_classes)
            raise ValueError(
                f"Diffusers image pipeline identity is inconsistent: runtime class {runtime_name!r} is tagged as "
                f"{adapter.pipeline_class}; allowed runtime classes: {allowed}."
            )

        mode = getattr(pipeline, "_modiff_image_mode")
        if not isinstance(mode, str) or not mode or mode != mode.strip() or mode not in adapter.modes:
            supported = ", ".join(adapter.mode_options)
            raise ValueError(
                f"{adapter.pipeline_class} carries invalid image mode {mode!r}. Supported modes: {supported}."
            )
        source = getattr(pipeline, "_modiff_image_source")
        if source not in _IMAGE_MODEL_SOURCES:
            raise ValueError("Diffusers image pipeline source tag must be canonical hub or local.")
        repository = getattr(pipeline, "_modiff_image_repo")
        if not isinstance(repository, str) or not repository or repository != repository.strip():
            raise ValueError("Diffusers image pipeline repository tag must be a nonblank canonical string.")
        revision = getattr(pipeline, "_modiff_image_revision")
        if source == "hub":
            if (
                not isinstance(revision, str)
                or revision != revision.lower()
                or not IMMUTABLE_HUB_REVISION.fullmatch(revision)
            ):
                raise ValueError("Diffusers image Hub pipeline revision tag must be a lowercase 40-character commit.")
            pin = catalog_repository_pin(repository)
            if pin is not None and revision != catalog_revision(repository):
                raise ValueError(
                    f"Diffusers image pipeline revision tag does not match the reviewed pin for {repository!r}."
                )
            managed = _managed_image_repositories()
            canonical_repository = managed.get(repository.casefold())
            if canonical_repository is not None:
                if repository != canonical_repository:
                    raise ValueError("Diffusers image pipeline repository tag is not canonically spelled.")
                if repository not in adapter.managed_repos:
                    raise ValueError(
                        f"Diffusers image pipeline repository {repository!r} is not compatible with "
                        f"{adapter.pipeline_class}."
                    )
        elif revision not in (None, ""):
            raise ValueError("A local Diffusers image pipeline must not carry a Hub revision tag.")
        if adapter.conditioning_kind is not None:
            conditioning_tags = {
                "kind": getattr(pipeline, "_modiff_conditioning_kind", None),
                "component class": getattr(pipeline, "_modiff_conditioning_component_class", None),
                "repository": getattr(pipeline, "_modiff_conditioning_repo", None),
                "revision": getattr(pipeline, "_modiff_conditioning_revision", None),
            }
            if conditioning_tags["kind"] != adapter.conditioning_kind:
                raise ValueError("Diffusers image conditioning kind does not match its reviewed adapter.")
            if conditioning_tags["component class"] != adapter.conditioning_component_class:
                raise ValueError("Diffusers image conditioning component class does not match its reviewed adapter.")
            conditioning_repo = conditioning_tags["repository"]
            conditioning_revision = conditioning_tags["revision"]
            if not isinstance(conditioning_repo, str) or not conditioning_repo:
                raise ValueError("Diffusers image conditioning repository tag is missing.")
            pin = catalog_repository_pin(conditioning_repo)
            if pin is not None:
                if conditioning_repo != pin.get("repo"):
                    raise ValueError("Diffusers image conditioning repository is not canonically spelled.")
                if conditioning_revision != catalog_revision(conditioning_repo):
                    raise ValueError("Diffusers image conditioning revision does not match its reviewed pin.")
            elif (
                not isinstance(conditioning_revision, str)
                or conditioning_revision != conditioning_revision.lower()
                or not IMMUTABLE_HUB_REVISION.fullmatch(conditioning_revision)
            ):
                raise ValueError("A custom image conditioning component requires an immutable commit tag.")
        return adapter

    candidates = [
        adapter for adapter in IMAGE_PIPELINE_ADAPTERS.values() if runtime_name in adapter.allowed_runtime_classes
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            f"Cannot recover an exact Diffusers image adapter from untagged runtime class {runtime_name!r}. "
            "Reconnect the pipeline through Load Diffusers Image Pipeline."
        )
    candidate_names = ", ".join(sorted(adapter.pipeline_class for adapter in candidates))
    raise ValueError(
        f"Cannot recover one exact Diffusers image adapter from untagged runtime class {runtime_name!r}: "
        f"{candidate_names}. Reconnect it through Load Diffusers Image Pipeline."
    )


def validate_image_action(pipeline: Any, action: str) -> ImagePipelineAdapter:
    accepted_modes = IMAGE_ACTION_MODES[action]
    adapter = _image_pipeline_adapter(pipeline)
    raw_mode = getattr(pipeline, "_modiff_image_mode", None)
    mode = raw_mode if isinstance(raw_mode, str) else ""

    if not mode:
        # Pipelines resident before mode tagging can be recovered only when all
        # adapter modes map to this one task node. Flux2 spans Generate and Edit,
        # so allowing either without its exact loader mode would be ambiguous.
        if set(adapter.modes).issubset(accepted_modes):
            return adapter
        supported_actions = [
            candidate
            for candidate, candidate_modes in IMAGE_ACTION_MODES.items()
            if set(adapter.modes).issubset(candidate_modes)
        ]
        suffix = f" Safe legacy action: {supported_actions[0]}." if len(supported_actions) == 1 else ""
        raise ValueError(
            f"{adapter.pipeline_class} is missing its exact loaded image mode, so {action} cannot run safely. "
            f"Reconnect it through Load Diffusers Image Pipeline.{suffix}"
        )
    if mode not in adapter.modes:
        supported = ", ".join(adapter.mode_options)
        raise ValueError(
            f"{adapter.pipeline_class} carries invalid image mode {mode!r}. Supported modes: {supported}."
        )
    if mode not in accepted_modes:
        required = ", ".join(accepted_modes)
        raise ValueError(
            f"{adapter.pipeline_class} was loaded for image mode {mode!r}; {action} requires one of: {required}."
        )
    return adapter


def _bounded_image_int(
    value: Any,
    *,
    field: str,
    default: int,
    minimum: int,
    maximum: int,
    step: int | None = None,
) -> int:
    raw = default if value is None else value
    if isinstance(raw, bool):
        raise ValueError(f"Diffusers image {field} must be an integer.")
    if isinstance(raw, int):
        parsed = raw
    else:
        try:
            numeric = float(raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Diffusers image {field} must be an integer.") from error
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"Diffusers image {field} must be a finite integer.")
        parsed = int(numeric)
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"Diffusers image {field} must be between {minimum} and {maximum}; received {parsed}.")
    if step is not None and (parsed - minimum) % step:
        raise ValueError(f"Diffusers image {field} must use increments of {step} from {minimum}; received {parsed}.")
    return parsed


def _bounded_image_float(
    value: Any,
    *,
    field: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = default if value is None else value
    if isinstance(raw, bool):
        raise ValueError(f"Diffusers image {field} must be a number.")
    try:
        parsed = float(raw)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Diffusers image {field} must be a number.") from error
    if not math.isfinite(parsed) or parsed < minimum or parsed > maximum:
        raise ValueError(
            f"Diffusers image {field} must be finite and between {minimum} and {maximum}; received {raw!r}."
        )
    return parsed


def _normalized_image_prompt(value: Any, *, field: str) -> str | list[str]:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        prompts = list(value)
        if not prompts or len(prompts) > 64 or any(not isinstance(item, str) for item in prompts):
            raise ValueError(f"Diffusers image {field} must be a string or a list of 1 to 64 strings.")
        return prompts
    raise ValueError(f"Diffusers image {field} must be a string or a list of 1 to 64 strings.")


def _image_media_extent(value: Any, *, field: str, require_pil: bool) -> tuple[int, int]:
    if isinstance(value, Image.Image):
        width, height = value.size
        sample_count = 1
    else:
        if require_pil:
            raise ValueError(f"Diffusers image {field} must be a PIL image loaded by an Image node.")
        torch_module = sys.modules.get("torch")
        tensor_type = getattr(torch_module, "Tensor", ()) if torch_module is not None else ()
        if not isinstance(value, np.ndarray) and not (tensor_type and isinstance(value, tensor_type)):
            raise ValueError(f"Diffusers image {field} must be a PIL image, NumPy array, or Torch tensor.")
        shape_value = getattr(value, "shape", None)
        try:
            shape = tuple(int(dimension) for dimension in shape_value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Diffusers image {field} must be a PIL image, NumPy array, or Torch tensor.") from error
        if not shape or any(dimension <= 0 for dimension in shape):
            raise ValueError(f"Diffusers image {field} dimensions must be positive and nonempty.")
        sample_count = shape[0] if len(shape) == 4 else 1
        image_shape = shape[1:] if len(shape) == 4 else shape
        if len(image_shape) == 2:
            height, width = image_shape
        elif len(image_shape) == 3 and image_shape[0] in {1, 3, 4}:
            _channels, height, width = image_shape
        elif len(image_shape) == 3 and image_shape[-1] in {1, 3, 4}:
            height, width, _channels = image_shape
        else:
            raise ValueError(f"Diffusers image {field} must have HW, CHW, HWC, NCHW, or NHWC image dimensions.")
    if width <= 0 or height <= 0:
        raise ValueError(f"Diffusers image {field} dimensions must be positive and nonempty.")
    if width > _MAX_IMAGE_INPUT_DIMENSION or height > _MAX_IMAGE_INPUT_DIMENSION:
        raise ValueError(
            f"Diffusers image {field} dimensions cannot exceed {_MAX_IMAGE_INPUT_DIMENSION} pixels per edge."
        )
    return sample_count, sample_count * width * height


def _validate_image_media(
    value: Any,
    *,
    field: str,
    max_items: int,
    max_pixels: int,
    require_pil: bool = False,
    require_single_value: bool = False,
) -> None:
    if value is None:
        raise ValueError(f"Diffusers image {field} is required.")
    if isinstance(value, (list, tuple)):
        if require_single_value:
            raise ValueError(f"Diffusers image {field} requires one PIL image, not a list.")
        items = list(value)
        if not items:
            raise ValueError(f"Diffusers image {field} cannot be an empty image list.")
    else:
        items = [value]

    total_items = 0
    total_pixels = 0
    for index, item in enumerate(items):
        sample_count, pixels = _image_media_extent(
            item,
            field=f"{field}[{index}]" if len(items) > 1 else field,
            require_pil=require_pil,
        )
        total_items += sample_count
        total_pixels += pixels
    if total_items > max_items:
        raise ValueError(f"Diffusers image {field} accepts at most {max_items} image(s); received {total_items}.")
    if total_pixels > max_pixels:
        raise ValueError(f"Diffusers image {field} exceeds the {max_pixels}-pixel cumulative input limit.")


def preflight_image_action(
    pipeline: Any,
    action: str,
    kwargs: dict[str, Any],
) -> tuple[ImagePipelineAdapter, dict[str, Any]]:
    """Validate the generic image contract before importing Torch or calling Diffusers."""

    if pipeline is None:
        raise ValueError("Diffusers image pipeline is required.")
    adapter = validate_image_action(pipeline, action)
    values = dict(kwargs)
    values["pipeline"] = pipeline
    values["prompt"] = _normalized_image_prompt(values.get("prompt"), field="prompt")
    values["negative_prompt"] = _normalized_image_prompt(
        values.get("negative_prompt"),
        field="negative_prompt",
    )
    values["width"] = _bounded_image_int(
        values.get("width"),
        field="width",
        default=1024,
        minimum=adapter.min_output_side,
        maximum=adapter.max_output_side,
        step=adapter.output_side_step,
    )
    values["height"] = _bounded_image_int(
        values.get("height"),
        field="height",
        default=1024,
        minimum=adapter.min_output_side,
        maximum=adapter.max_output_side,
        step=adapter.output_side_step,
    )
    output_pixels = values["width"] * values["height"]
    if output_pixels > adapter.max_output_pixels:
        raise ValueError(
            f"{adapter.pipeline_class} output cannot exceed {adapter.max_output_pixels} pixels; "
            f"received {values['width']}x{values['height']} ({output_pixels} pixels)."
        )
    values["seed"] = _bounded_image_int(values.get("seed"), field="seed", default=0, minimum=0, maximum=4294967295)
    values["num_inference_steps"] = _bounded_image_int(
        values.get("num_inference_steps"),
        field="num_inference_steps",
        default=4,
        minimum=1,
        maximum=adapter.max_inference_steps,
    )
    values["guidance_scale"] = _bounded_image_float(
        values.get("guidance_scale"), field="guidance_scale", default=0.0, minimum=0.0, maximum=20.0
    )
    if adapter.fixed_guidance_scale is not None and values["guidance_scale"] != adapter.fixed_guidance_scale:
        raise ValueError(
            f"{adapter.pipeline_class} requires guidance_scale={adapter.fixed_guidance_scale:g}."
        )
    values["image_guidance_scale"] = _bounded_image_float(
        values.get("image_guidance_scale"),
        field="image_guidance_scale",
        default=1.5,
        minimum=adapter.minimum_image_guidance_scale,
        maximum=20.0,
    )
    values["pag_scale"] = _bounded_image_float(
        values.get("pag_scale"), field="pag_scale", default=3.0, minimum=0.0, maximum=20.0
    )
    values["pag_adaptive_scale"] = _bounded_image_float(
        values.get("pag_adaptive_scale"),
        field="pag_adaptive_scale",
        default=0.0,
        minimum=0.0,
        maximum=20.0,
    )
    values["strength"] = _bounded_image_float(
        values.get("strength"), field="strength", default=0.8, minimum=0.0, maximum=1.0
    )
    values["padding_mask_crop"] = _bounded_image_int(
        values.get("padding_mask_crop"),
        field="padding_mask_crop",
        default=0,
        minimum=0,
        maximum=512,
        step=8,
    )
    values["max_sequence_length"] = _bounded_image_int(
        values.get("max_sequence_length"),
        field="max_sequence_length",
        default=256,
        minimum=1,
        maximum=adapter.max_sequence_length,
    )
    values["reference_strength"] = _bounded_image_float(
        values.get("reference_strength"),
        field="reference_strength",
        default=1.0,
        minimum=0.0,
        maximum=1.0,
    )
    values["conditioning_scale"] = _bounded_image_float(
        values.get("conditioning_scale"),
        field="conditioning_scale",
        default=1.0,
        minimum=0.0,
        maximum=2.0,
    )

    output_type = (
        "pil" if "output_type" not in values or values.get("output_type") is None else values.get("output_type")
    )
    allowed_output_types = {"pil"} if action == "Inpaint" else {"pil", "np", "pt"}
    if not isinstance(output_type, str) or output_type not in allowed_output_types:
        allowed = ", ".join(sorted(allowed_output_types))
        raise ValueError(f"Diffusers image {action} output_type must be exactly one of: {allowed}.")
    values["output_type"] = output_type

    if action == "Edit":
        mode = getattr(pipeline, "_modiff_image_mode", None)
        max_references = adapter.max_reference_images if mode in (None, "multi_image_reference_edit") else 1
        _validate_image_media(
            values.get("image"),
            field="Edit image",
            max_items=max_references,
            max_pixels=adapter.max_reference_pixels,
        )
    elif action == "Inpaint":
        _validate_image_media(
            values.get("image"),
            field="Inpaint source",
            max_items=1,
            max_pixels=adapter.max_reference_pixels,
            require_pil=True,
            require_single_value=True,
        )
        _validate_image_media(
            values.get("mask_image"),
            field="Inpaint mask",
            max_items=1,
            max_pixels=adapter.max_reference_pixels,
            require_pil=True,
            require_single_value=True,
        )
    elif action == "ControlGenerate":
        _validate_image_media(
            values.get("control_image"),
            field="Control image",
            max_items=1,
            max_pixels=adapter.max_reference_pixels,
        )
    return adapter, values


def preflight_unconditional_action(
    pipeline: Any,
    kwargs: dict[str, Any],
) -> tuple[ImagePipelineAdapter, dict[str, Any]]:
    """Validate model-neutral unconditional sampling inputs before Torch execution."""

    if pipeline is None:
        raise ValueError("Diffusers image pipeline is required.")
    adapter = validate_image_action(pipeline, "UnconditionalGenerate")
    values = dict(kwargs)
    values["pipeline"] = pipeline
    values["batch_size"] = _bounded_image_int(
        values.get("batch_size"), field="batch_size", default=1, minimum=1, maximum=16
    )
    values["seed"] = _bounded_image_int(
        values.get("seed"), field="seed", default=0, minimum=0, maximum=4294967295
    )
    values["num_inference_steps"] = _bounded_image_int(
        values.get("num_inference_steps"),
        field="num_inference_steps",
        default=50,
        minimum=1,
        maximum=1000,
    )
    values["eta"] = _bounded_image_float(
        values.get("eta"), field="eta", default=0.0, minimum=0.0, maximum=1.0
    )
    values["class_label"] = _bounded_image_int(
        values.get("class_label"), field="class_label", default=-1, minimum=-1, maximum=999
    )
    output_type = "pil" if values.get("output_type") is None else values.get("output_type")
    if output_type not in {"pil", "np"}:
        raise ValueError("Diffusers image UnconditionalGenerate output_type must be exactly one of: np, pil.")
    values["output_type"] = output_type
    return adapter, values


def preflight_prediction_map_action(
    pipeline: Any,
    kwargs: dict[str, Any],
) -> tuple[ImagePipelineAdapter, dict[str, Any]]:
    """Validate one generic perception request before importing Torch or calling Diffusers."""

    if pipeline is None:
        raise ValueError("Diffusers image pipeline is required.")
    adapter = validate_image_action(pipeline, "PredictMap")
    values = dict(kwargs)
    values["pipeline"] = pipeline
    _validate_image_media(
        values.get("image"),
        field="prediction source",
        max_items=1,
        max_pixels=adapter.max_reference_pixels,
    )
    image = values.get("image")
    if isinstance(image, (list, tuple)):
        if len(image) != 1:
            raise ValueError("Diffusers image prediction source requires exactly one image.")
        image = image[0]
    values["image"] = image
    values["seed"] = _bounded_image_int(
        values.get("seed"), field="seed", default=0, minimum=0, maximum=4294967295
    )
    values["num_inference_steps"] = _bounded_image_int(
        values.get("num_inference_steps"),
        field="num_inference_steps",
        default=1,
        minimum=1,
        maximum=50,
    )
    values["processing_resolution"] = _bounded_image_int(
        values.get("processing_resolution"),
        field="processing_resolution",
        default=768,
        minimum=64,
        maximum=2048,
        step=8,
    )
    match_input_resolution = values.get("match_input_resolution", True)
    if type(match_input_resolution) is not bool:
        raise ValueError("Diffusers image match_input_resolution must be a boolean.")
    values["match_input_resolution"] = match_input_resolution
    prediction_kind = values.get("prediction_kind", "depth")
    if prediction_kind != "depth":
        raise ValueError("Diffusers image prediction_kind must be exactly depth for this pipeline mode.")
    values["prediction_kind"] = prediction_kind
    return adapter, values


def normalize_prediction_map(prediction: Any, *, kind: str) -> tuple[dict[str, Any], list[Image.Image]]:
    """Normalize one perception result into the versioned MoDiff prediction-map contract."""

    if hasattr(prediction, "detach"):
        prediction = prediction.detach()
    if hasattr(prediction, "float"):
        prediction = prediction.float()
    if hasattr(prediction, "cpu"):
        prediction = prediction.cpu()
    if hasattr(prediction, "numpy"):
        prediction = prediction.numpy()
    try:
        array = np.asarray(prediction, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Diffusers prediction output must be a numeric array.") from error

    if array.ndim == 2:
        array = array[np.newaxis, ..., np.newaxis]
    elif array.ndim == 3:
        array = array[np.newaxis, ...] if array.shape[-1] == 1 else array[..., np.newaxis]
    elif array.ndim == 4 and array.shape[-1] != 1 and array.shape[1] == 1:
        array = np.transpose(array, (0, 2, 3, 1))
    if array.ndim != 4 or array.shape[0] != 1 or array.shape[-1] != 1:
        raise ValueError("Diffusers prediction output must contain exactly one single-channel map.")

    _batch, height, width, _channels = array.shape
    if height <= 0 or width <= 0:
        raise ValueError("Diffusers prediction output dimensions must be positive and nonempty.")
    if width > _MAX_IMAGE_INPUT_DIMENSION or height > _MAX_IMAGE_INPUT_DIMENSION:
        raise ValueError(
            f"Diffusers prediction output dimensions cannot exceed {_MAX_IMAGE_INPUT_DIMENSION} pixels per edge."
        )
    if width * height > _MAX_IMAGE_INPUT_PIXELS:
        raise ValueError(f"Diffusers prediction output exceeds the {_MAX_IMAGE_INPUT_PIXELS}-pixel limit.")
    if not np.isfinite(array).all():
        raise ValueError("Diffusers prediction output must contain only finite values.")
    tolerance = 1e-5
    if float(array.min()) < -tolerance or float(array.max()) > 1.0 + tolerance:
        raise ValueError("Diffusers prediction output must stay inside the normalized [0, 1] value range.")
    array = np.ascontiguousarray(np.clip(array, 0.0, 1.0), dtype=np.float32)
    preview = Image.fromarray(np.rint(array[0, ..., 0] * 255.0).astype(np.uint8), mode="L").convert("RGB")
    prediction_map = {
        "schemaVersion": 1,
        "kind": kind,
        "semantics": "relative_depth",
        "layout": "NHWC",
        "dtype": "float32",
        "shape": [1, height, width, 1],
        "valueRange": [0.0, 1.0],
        "nearValue": 0.0,
        "farValue": 1.0,
        "width": width,
        "height": height,
        "prediction": array,
    }
    return prediction_map, [preview]


def output_image_dimensions(images: Any, output_type: str = "pil") -> tuple[int | None, int | None]:
    """Return width/height for Diffusers PIL, NumPy, or Torch outputs."""

    first = images[0] if isinstance(images, (list, tuple)) and images else images
    size = getattr(first, "size", None)
    if isinstance(size, (tuple, list)) and len(size) >= 2:
        return int(size[0]), int(size[1])

    shape_value = getattr(first, "shape", None)
    if shape_value is None:
        shape_value = getattr(images, "shape", None)
    try:
        shape = tuple(int(value) for value in shape_value)
    except (TypeError, ValueError):
        return None, None
    if len(shape) < 2:
        return None, None
    if str(output_type).lower() == "pt":
        height, width = shape[-2], shape[-1]
    elif len(shape) >= 4:
        height, width = shape[-3], shape[-2]
    else:
        height, width = shape[0], shape[1]
    return width, height


def ensure_single_image(value: Any, field_name: str) -> Image.Image:
    if value in (None, ""):
        raise ValueError(f"Outpaint Canvas requires a {field_name}.")
    if isinstance(value, list):
        images = [item for item in value if item not in (None, "")]
        if len(images) != 1:
            raise ValueError(f"Outpaint Canvas requires exactly one {field_name}; received {len(images)}.")
        value = images[0]
    if not isinstance(value, Image.Image):
        raise ValueError(f"Outpaint Canvas {field_name} must be a PIL image loaded by the Image Load node.")
    return value


def _int_in_range(value: Any, fallback: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(minimum, min(maximum, parsed))


def _float_in_range(value: Any, fallback: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(minimum, min(maximum, parsed))


def _placement_offset(available: int, start_margin: int, end_margin: int) -> int:
    if available <= 0:
        return 0
    requested = start_margin + end_margin
    if requested <= 0:
        return available // 2
    if requested > available:
        return int(round((start_margin / requested) * available))
    return min(start_margin, available)


def _outpaint_source_size(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    margins: tuple[int, int, int, int],
) -> tuple[int, int]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    left, right, top, bottom = margins
    available_width = max(1, target_width - left - right)
    available_height = max(1, target_height - top - bottom)
    scale = min(
        target_width / source_width,
        target_height / source_height,
        available_width / source_width,
        available_height / source_height,
        1.0,
    )
    return (
        max(1, int(round(source_width * scale))),
        max(1, int(round(source_height * scale))),
    )


class OutpaintCanvas(NodeBase):
    """Prepare a model-neutral expanded canvas and white-generate boundary mask."""

    label = "Outpaint Canvas"
    category = "Diffusers Image"
    resizable = True
    params = {
        "image": {"label": "Source image", "display": "input", "type": "image"},
        "width": {"label": "Canvas width", "type": "int", "default": 1344, "min": 64, "max": 2048, "step": 16},
        "height": {"label": "Canvas height", "type": "int", "default": 768, "min": 64, "max": 2048, "step": 16},
        "left": {"label": "Left margin", "type": "int", "default": 256, "min": 0, "max": 2048, "step": 16},
        "right": {"label": "Right margin", "type": "int", "default": 256, "min": 0, "max": 2048, "step": 16},
        "top": {"label": "Top margin", "type": "int", "default": 0, "min": 0, "max": 2048, "step": 16},
        "bottom": {"label": "Bottom margin", "type": "int", "default": 0, "min": 0, "max": 2048, "step": 16},
        "overlap": {"label": "Seam overlap", "type": "int", "default": 24, "min": 0, "max": 256, "step": 4},
        "feather": {"label": "Mask feather", "type": "float", "default": 8.0, "min": 0, "max": 128, "step": 1},
        "fill_color": {"label": "Fill color", "type": "string", "default": "#000000"},
        "canvas": {"label": "Canvas", "display": "output", "type": "image"},
        "mask_image": {"label": "Mask image", "display": "output", "type": "image"},
        "width_out": {"label": "Width", "display": "output", "type": "int"},
        "height_out": {"label": "Height", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        source = ensure_single_image(kwargs.get("image"), "source image")
        target_width = _int_in_range(kwargs.get("width"), 1344, 64, 2048)
        target_height = _int_in_range(kwargs.get("height"), 768, 64, 2048)
        left = _int_in_range(kwargs.get("left"), 256, 0, target_width)
        right = _int_in_range(kwargs.get("right"), 256, 0, target_width)
        top = _int_in_range(kwargs.get("top"), 0, 0, target_height)
        bottom = _int_in_range(kwargs.get("bottom"), 0, 0, target_height)
        overlap = _int_in_range(kwargs.get("overlap"), 24, 0, 256)
        feather = _float_in_range(kwargs.get("feather"), 8.0, 0.0, 128.0)
        try:
            fill_color = ImageColor.getrgb(str(kwargs.get("fill_color") or "#000000"))[:3]
        except ValueError as exc:
            raise ValueError("Outpaint Canvas fill color must be a CSS color name or hex color.") from exc

        source_rgba = source.convert("RGBA")
        pasted_width, pasted_height = _outpaint_source_size(
            source_rgba.size,
            (target_width, target_height),
            (left, right, top, bottom),
        )
        if (pasted_width, pasted_height) != source_rgba.size:
            source_rgba = source_rgba.resize((pasted_width, pasted_height), Image.Resampling.LANCZOS)

        offset_x = _placement_offset(target_width - pasted_width, left, right)
        offset_y = _placement_offset(target_height - pasted_height, top, bottom)
        canvas = Image.new("RGB", (target_width, target_height), fill_color)
        canvas.paste(source_rgba, (offset_x, offset_y), source_rgba)
        mask = Image.new("L", (target_width, target_height), 255)
        keep_left = min(target_width, max(0, offset_x + overlap))
        keep_top = min(target_height, max(0, offset_y + overlap))
        keep_right = min(target_width, max(0, offset_x + pasted_width - overlap))
        keep_bottom = min(target_height, max(0, offset_y + pasted_height - overlap))
        if keep_right > keep_left and keep_bottom > keep_top:
            ImageDraw.Draw(mask).rectangle((keep_left, keep_top, keep_right, keep_bottom), fill=0)
        if feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
        return {
            "canvas": canvas,
            "mask_image": mask,
            "width_out": target_width,
            "height_out": target_height,
        }


def composite_masked_pil_outputs(outputs: Any, source: Any, mask: Any) -> list[Image.Image]:
    """Keep generic inpaint outputs byte-stable outside a white-generate mask."""
    source_image = source if isinstance(source, Image.Image) else None
    mask_image = mask if isinstance(mask, Image.Image) else None
    generated = (
        [outputs]
        if isinstance(outputs, Image.Image)
        else list(outputs or [])
        if isinstance(outputs, (list, tuple))
        else []
    )
    if (
        source_image is None
        or mask_image is None
        or not generated
        or any(not isinstance(item, Image.Image) for item in generated)
    ):
        raise ValueError(
            "Diffusers image inpaint requires PIL source, mask, and output images for mask-safe compositing."
        )

    composited = []
    for image in generated:
        target_size = image.size
        normalized_source = source_image.convert("RGB")
        if normalized_source.size != target_size:
            normalized_source = normalized_source.resize(target_size, Image.Resampling.LANCZOS)
        normalized_mask = mask_image.convert("L")
        if normalized_mask.size != target_size:
            normalized_mask = normalized_mask.resize(target_size, Image.Resampling.LANCZOS)
        composited.append(Image.composite(image.convert("RGB"), normalized_source, normalized_mask))
    return composited


def normalize_component_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        raw_items = [str(item).strip() for item in value]
    else:
        raw_items = [str(value).strip()]
    return [item for item in raw_items if item in QUANT_COMPONENTS]


def supports_arg(pipeline: Any, arg_name: str) -> bool:
    try:
        return arg_name in inspect.signature(pipeline.__call__).parameters
    except (TypeError, ValueError):
        return False


def pipeline_class_from_name(name: str):
    import diffusers

    pipeline_class = getattr(diffusers, name, None)
    if pipeline_class is None:
        raise ValueError(f"Diffusers does not expose pipeline class {name}. Update diffusers or choose another class.")
    return pipeline_class


def quant_config_for(method: str, dtype: Any, modules_to_not_convert: list[str] | None = None):
    excluded = list(dict.fromkeys(modules_to_not_convert or [])) or None
    if method == "bnb_4bit":
        from diffusers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            llm_int8_skip_modules=excluded,
        )
    if method == "bnb_8bit":
        from diffusers import BitsAndBytesConfig

        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=excluded)
    if method == "quanto_float8":
        from diffusers import QuantoConfig

        return QuantoConfig(weights_dtype="float8", modules_to_not_convert=excluded)
    if method == "quanto_int8":
        from diffusers import QuantoConfig

        return QuantoConfig(weights_dtype="int8", modules_to_not_convert=excluded)
    if method == "torchao_float8":
        from diffusers import TorchAoConfig

        try:
            from torchao.quantization import Float8WeightOnlyConfig
        except ImportError as exc:
            raise RuntimeError(
                "TorchAO FP8 requires the optional torchao quantization package. "
                "Install MoDiff's quantization dependencies and retry."
            ) from exc

        return TorchAoConfig(
            quant_type=Float8WeightOnlyConfig(),
            modules_to_not_convert=excluded,
        )
    if method == "torchao_int8_weight_only":
        from diffusers import TorchAoConfig

        try:
            from torchao.quantization import Int8WeightOnlyConfig
        except ImportError as exc:
            raise RuntimeError(
                "TorchAO INT8 weight-only quantization requires the optional torchao package. "
                "Install MoDiff's quantization dependencies and retry."
            ) from exc

        return TorchAoConfig(
            quant_type=Int8WeightOnlyConfig(),
            modules_to_not_convert=excluded,
        )
    if method in {"torchao_mxfp8", "torchao_nvfp4"}:
        from diffusers import TorchAoConfig

        try:
            if method == "torchao_mxfp8":
                from torchao.prototype.mx_formats.inference_workflow import MXDynamicActivationMXWeightConfig
                from torchao.quantization.quantize_.common import KernelPreference
                import torch

                config = MXDynamicActivationMXWeightConfig(
                    activation_dtype=torch.float8_e4m3fn,
                    weight_dtype=torch.float8_e4m3fn,
                    kernel_preference=KernelPreference.AUTO,
                )
            else:
                from torchao.prototype.mx_formats.inference_workflow import (
                    NVFP4DynamicActivationNVFP4WeightConfig,
                )

                config = NVFP4DynamicActivationNVFP4WeightConfig(
                    use_dynamic_per_tensor_scale=True,
                    use_triton_kernel=True,
                )
        except ImportError as exc:
            raise RuntimeError(
                f"{method.removeprefix('torchao_').upper()} artifact creation requires a compatible "
                "Blackwell PyTorch, TorchAO, and kernel environment."
            ) from exc
        return TorchAoConfig(quant_type=config, modules_to_not_convert=excluded)
    return None


def build_pipeline_quantization_config(method: str, components: list[str], dtype: Any):
    if method == "none" or not components:
        return None
    config = quant_config_for(method, dtype)
    if config is None:
        return None
    from diffusers.quantizers import PipelineQuantizationConfig

    return PipelineQuantizationConfig(quant_mapping={component: config for component in components})


def coerce_pipeline_quantization_config(quant_config: Any):
    """Normalize graph-provided component mappings to Diffusers' public config."""

    if not quant_config:
        return None
    if isinstance(quant_config, str):
        if not quant_config.strip():
            return None
        raise ValueError("Quantization config must be connected to a Quantization Config node output.")
    from diffusers.quantizers import PipelineQuantizationConfig

    if isinstance(quant_config, PipelineQuantizationConfig):
        return quant_config
    if isinstance(quant_config, dict):
        return PipelineQuantizationConfig(quant_mapping=quant_config)
    raise ValueError("Quantization config must be a Diffusers PipelineQuantizationConfig or component mapping.")


def build_qwen_pipeline_quantization_config(
    *,
    components: list[str],
    quantization_mode: str,
    compute_dtype: Any,
    quant_type: str = "nf4",
    double_quant: bool = True,
):
    """Build component-correct BnB configs for Qwen's Diffusers and Transformers parts."""

    if quantization_mode != "bnb_4bit" or not components:
        return None
    from importlib.util import find_spec

    if find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "BitsAndBytes 4-bit quantization is not installed. Install the CUDA quantization extra or choose another mode."
        )
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    from diffusers.quantizers import PipelineQuantizationConfig
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

    quant_mapping = {}
    if "transformer" in components:
        quant_mapping["transformer"] = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bool(double_quant),
        )
    if "text_encoder" in components:
        quant_mapping["text_encoder"] = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bool(double_quant),
        )
    return PipelineQuantizationConfig(quant_mapping=quant_mapping) if quant_mapping else None


def add_progress_callback(node: NodeBase, pipeline: Any, call_kwargs: dict[str, Any], steps: int):
    node.progress(
        0,
        phase="denoising",
        message=f"Denoising 0/{steps}",
        current_step=0,
        total_steps=steps,
        elapsed_seconds=0.0,
    )
    if not supports_arg(pipeline, "callback_on_step_end"):
        return

    def callback(pipe, step_index, timestep, callback_kwargs):
        # NodeBase owns the common cancellation contract. Propagating it here
        # makes /stop interrupt generic Diffusers pipelines at the next step,
        # just like the established video/audio nodes.
        return node.pipe_callback(pipe, step_index, timestep, callback_kwargs) or callback_kwargs

    call_kwargs["callback_on_step_end"] = callback
    if supports_arg(pipeline, "callback_on_step_end_tensor_inputs"):
        call_kwargs["callback_on_step_end_tensor_inputs"] = []


def prepare_reference_images(image: Any, adapter: ImagePipelineAdapter) -> Any:
    """Translate the generic multi-reference contract for a pipeline adapter."""
    if not isinstance(image, (list, tuple)) or len(image) <= 1:
        return image[0] if isinstance(image, (list, tuple)) and image else image
    if adapter.multi_image_strategy != "stitch_horizontal":
        return image if isinstance(image, list) else list(image)

    from PIL import Image

    if not all(isinstance(item, Image.Image) for item in image):
        raise ValueError("Horizontal multi-reference stitching currently requires PIL image inputs.")
    converted = [item.convert("RGB") for item in image]
    target_height = max(item.height for item in converted)
    resized = [
        item
        if item.height == target_height
        else item.resize(
            (max(1, round(item.width * target_height / item.height)), target_height), Image.Resampling.LANCZOS
        )
        for item in converted
    ]
    canvas = Image.new("RGB", (sum(item.width for item in resized), target_height))
    left = 0
    for item in resized:
        canvas.paste(item, (left, 0))
        left += item.width
    return canvas


class FluxReduxPipelineBundle:
    """Callable adapter combining the official Redux prior with FLUX.1-dev."""

    def __init__(self, prior: Any, base: Any):
        self.prior = prior
        self.base = base

    @property
    def device(self):
        return getattr(self.base, "device", "cpu")

    @property
    def _execution_device(self):
        return getattr(self.base, "_execution_device", self.device)

    def __call__(
        self,
        *,
        image,
        prompt="",
        negative_prompt=None,
        width=None,
        height=None,
        num_inference_steps=28,
        guidance_scale=3.5,
        strength=None,
        padding_mask_crop=None,
        max_sequence_length=512,
        generator=None,
        output_type="pil",
        return_dict=True,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=None,
        reference_strength=1.0,
    ):
        # FluxPriorReduxPipeline produces the exact reference/text embeddings
        # consumed by FluxPipeline. For multiple images, current Diffusers
        # applies the per-image scales and returns one upstream weighted sum.
        prior_kwargs = {"image": image, "prompt": prompt or None, "return_dict": True}
        if isinstance(image, (list, tuple)) and len(image) > 1:
            secondary_strength = float(reference_strength)
            if not 0.0 <= secondary_strength <= 1.0:
                raise ValueError("reference_strength must be between 0 and 1.")
            reference_scales = [1.0, *([secondary_strength] * (len(image) - 1))]
            prior_kwargs["prompt_embeds_scale"] = reference_scales
            prior_kwargs["pooled_prompt_embeds_scale"] = reference_scales
        prior_output = self.prior(**prior_kwargs)
        prompt_embeds = prior_output.prompt_embeds
        pooled_prompt_embeds = prior_output.pooled_prompt_embeds
        base_kwargs = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "output_type": output_type,
            "return_dict": return_dict,
            "max_sequence_length": max_sequence_length,
        }
        for key, value in (("width", width), ("height", height)):
            if value is not None:
                base_kwargs[key] = value
        if callback_on_step_end is not None:
            base_kwargs["callback_on_step_end"] = callback_on_step_end
        if callback_on_step_end_tensor_inputs is not None:
            base_kwargs["callback_on_step_end_tensor_inputs"] = callback_on_step_end_tensor_inputs
        return self.base(**base_kwargs)


class LoadPipeline(NodeBase):
    """Load a generic Diffusers image pipeline."""

    label = "Load Diffusers Image Pipeline"
    category = "Diffusers Image"
    resizable = True
    # Flux2KleinPipeline uses one resident pipeline for generation and edit
    # calls. Mode validates the requested operation but does not participate in
    # from_pretrained(), so changing it must not force another 13-minute load.
    cache_ignored_params = frozenset({"mode"})
    params = {
        "pipeline": {
            "label": "Pipeline",
            "display": "output",
            "type": "image_diffusion_pipeline",
            "signal": {
                "direction": "output",
                "origin": "pipeline_class",
                "value": DEFAULT_IMAGE_PIPELINE_CONTRACT,
            },
        },
        "model_id": {
            "label": "Model",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": FLUX_SCHNELL_REPO},
            "fieldOptions": {
                "noValidation": True,
                "sources": ["hub", "local"],
                "filter": {
                    "hub": {"className": ["FluxPipeline"]},
                    "local": {"className": ["FluxPipeline"]},
                },
            },
            "onChange": "update_pipeline_contract",
        },
        "pipeline_class": {
            "label": "Pipeline Class",
            "type": "string",
            "options": IMAGE_PIPELINE_CLASSES,
            "default": "FluxPipeline",
            "fieldOptions": {"noValidation": True},
            "onChange": "update_pipeline_contract",
        },
        "mode": {
            "label": "Mode",
            "type": "string",
            "options": IMAGE_PIPELINE_MODE_OPTIONS,
            "default": "text_to_image",
            "fieldOptions": {"noValidation": True},
            "onChange": "update_pipeline_contract",
        },
        "revision": {"label": "Revision", "type": "string", "default": ""},
        "conditioning_kind": {
            "label": "Conditioning Kind",
            "type": "string",
            "options": ["none", "controlnet", "t2i_adapter"],
            "default": "none",
            "fieldOptions": {"noValidation": True},
        },
        "conditioning_model_id": {
            "label": "Conditioning Model",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": SD15_CONTROLNET_CANNY_REPO},
            "fieldOptions": {
                "noValidation": True,
                "sources": ["hub"],
                "filter": {"hub": {"className": ["ControlNetModel"]}},
            },
        },
        "conditioning_revision": {
            "label": "Conditioning Revision",
            "type": "string",
            "default": "",
        },
        "dtype": {
            "label": "DType",
            "type": "string",
            "options": ["float32", "float16", "bfloat16"],
            "default": "bfloat16",
        },
        "device": {"label": "Device", "type": "string", "options": DEVICE_OPTIONS, "default": DEFAULT_DEVICE},
        "quantization_mode": {
            "label": "Quantization",
            "type": "string",
            "options": [
                "none",
                "bnb_4bit",
                "bnb_8bit",
                "quanto_float8",
                "quanto_int8",
                "torchao_float8",
                "torchao_int8_weight_only",
            ],
            "default": "none",
        },
        "quantized_components": {
            "label": "Quantized Components",
            "type": "string",
            "display": "select",
            "options": QUANT_COMPONENTS,
            "fieldOptions": {"multiple": True},
            "default": [],
        },
        "execution_recipe": {
            "label": "Execution Recipe",
            "display": "input",
            "type": "diffusers_execution_recipe",
        },
        "device_map": {
            "label": "Device Map",
            "type": "string",
            "options": ["none", "cuda", "auto", "balanced", "balanced_low_0", "cpu"],
            "default": "none",
        },
        "auto_offload": {"label": "Auto offload", "type": "bool", "default": True},
        "offload_mode": offload_mode_param(modes=DIFFUSERS_IMAGE_OFFLOAD_MODES),
        "enable_vae_slicing": {"label": "VAE slicing", "type": "bool", "default": True},
        "enable_vae_tiling": {"label": "VAE tiling", "type": "bool", "default": True},
        "low_cpu_mem_usage": {"label": "Low CPU memory", "type": "bool", "default": True},
        "resolved_artifact": {"label": "Resolved Artifact", "display": "output", "type": "string"},
    }

    @staticmethod
    def _validate_mode(pipeline_class_name: str, requested_mode: Any):
        adapter = get_image_pipeline_adapter(pipeline_class_name)
        mode = _loader_image_mode({"mode": requested_mode}, adapter)
        return adapter, mode

    def __call__(self, **kwargs):
        adapter = _loader_image_pipeline_adapter(kwargs)
        requested_mode = _loader_image_mode(kwargs, adapter)
        values = dict(kwargs)
        values["pipeline_class"] = adapter.pipeline_class
        values["mode"] = requested_mode
        values["model_id"] = resolve_image_model_selection(adapter, values.get("model_id"))
        values["revision"] = resolve_image_pipeline_revision(values["model_id"], values.get("revision"))
        conditioning_selection, conditioning_revision = resolve_image_conditioning_selection(
            adapter,
            values.get("conditioning_kind"),
            values.get("conditioning_model_id"),
            values.get("conditioning_revision"),
        )
        if adapter.conditioning_kind is not None:
            values["conditioning_kind"] = adapter.conditioning_kind
            values["conditioning_model_id"] = conditioning_selection
            values["conditioning_revision"] = conditioning_revision
        else:
            values["conditioning_kind"] = "none"
            values["conditioning_model_id"] = ""
            values["conditioning_revision"] = ""
        result = super().__call__(**values)
        pipeline = result.get("pipeline") if isinstance(result, dict) else None
        if pipeline is not None:
            _tag_image_pipeline(
                pipeline,
                adapter,
                requested_mode,
                repo_value(values["model_id"]),
                values["model_id"]["source"],
                values["revision"],
                conditioning_repo=(
                    repo_value(values["conditioning_model_id"])
                    if conditioning_selection is not None
                    else None
                ),
                conditioning_revision=conditioning_revision,
            )
        return result

    def update_pipeline_contract(self, values, ref):
        if not isinstance(values, dict):
            raise ValueError("Diffusers image loader values must be an object.")
        adapter = _loader_image_pipeline_adapter(values)
        requested_mode = values.get("mode")
        if not isinstance(requested_mode, str) or not requested_mode or requested_mode != requested_mode.strip():
            raise ValueError("A registered Diffusers image mode is required.")
        signal_origin = ref.get("key") if isinstance(ref, dict) else ref
        if requested_mode in adapter.modes:
            selected_mode = requested_mode
        elif signal_origin == "pipeline_class" and requested_mode in IMAGE_PIPELINE_MODE_OPTIONS:
            selected_mode = adapter.mode_options[0]
        else:
            _loader_image_mode(values, adapter)
            raise AssertionError("unreachable")
        current_selection = values.get("model_id")
        resolved_selection = resolve_image_model_selection(adapter, current_selection)
        raw_revision = _normalize_image_revision(values.get("revision"))
        if resolved_selection.get("source") == "local":
            resolved_revision = ""
        else:
            selected_repo = resolved_selection["value"]
            reviewed_revision = catalog_revision(selected_repo)
            if reviewed_revision is not None:
                # A managed repository always owns its catalog pin. This also
                # replaces a stale custom pin when a class change selects the
                # class's reviewed default repository.
                resolved_revision = reviewed_revision
            elif signal_origin == "model_id":
                # A field action has no trustworthy previous-repository value,
                # so a custom repository selection cannot inherit the visible
                # pin from whichever repository was selected before it.
                resolved_revision = ""
            else:
                selection_was_replaced = False
                if isinstance(current_selection, dict):
                    raw_source = current_selection.get("source")
                    raw_repo = current_selection.get("value")
                    selection_was_replaced = not (
                        isinstance(raw_source, str)
                        and raw_source.strip().casefold() == resolved_selection["source"]
                        and isinstance(raw_repo, str)
                        and raw_repo.strip().casefold() == selected_repo.casefold()
                    )
                elif isinstance(current_selection, str):
                    selection_was_replaced = current_selection.strip().casefold() != selected_repo.casefold()
                else:
                    selection_was_replaced = True
                resolved_revision = (
                    "" if selection_was_replaced else resolve_image_pipeline_revision(resolved_selection, raw_revision)
                )

        self.set_field_params(
            "mode",
            {"options": list(adapter.mode_options), "default": adapter.mode_options[0]},
        )
        self.set_field_params("model_id", {"fieldOptions": image_model_field_options(adapter)})
        if selected_mode != requested_mode:
            self.set_field_value({"mode": selected_mode})
        if resolved_selection != current_selection:
            self.set_field_value({"model_id": resolved_selection})
        if resolved_revision != raw_revision:
            self.set_field_value({"revision": resolved_revision})
        self.set_field_params(
            "pipeline",
            {
                "signal": {
                    "direction": "output",
                    "origin": str(signal_origin or "pipeline_class"),
                    "value": image_pipeline_contract(adapter, selected_mode),
                }
            },
        )

    def prepare_for_workflow_reuse(self):
        """Restore a resident image pipeline before another graph adopts it."""
        pipeline = self.output.get("pipeline")
        unload_lora_weights = getattr(pipeline, "unload_lora_weights", None)
        if callable(unload_lora_weights):
            unload_lora_weights()

    def execute(self, **kwargs):
        from modules.DiffusersRuntime.main import (
            apply_execution_recipe_to_pipeline,
            assert_runtime_quantization_full_residency,
            enable_parallel_weight_loading,
        )

        execution_recipe = kwargs.get("execution_recipe")
        if execution_recipe is None:
            execution_recipe = {}
        if not isinstance(execution_recipe, dict):
            raise TypeError("Execution Recipe must come from a Diffusers Execution Recipe node.")
        adapter = _loader_image_pipeline_adapter(kwargs)
        pipeline_class_name = adapter.pipeline_class
        requested_mode = _loader_image_mode(kwargs, adapter)
        model_selection = resolve_image_model_selection(adapter, kwargs.get("model_id"))
        model_id = repo_value(model_selection)
        model_source = model_selection["source"]
        dtype = str_to_dtype(kwargs.get("dtype") or "bfloat16")
        device = execution_recipe.get("device") or kwargs.get("device") or DEFAULT_DEVICE
        revision = resolve_image_pipeline_revision(model_selection, kwargs.get("revision"))
        conditioning_selection, conditioning_revision = resolve_image_conditioning_selection(
            adapter,
            kwargs.get("conditioning_kind"),
            kwargs.get("conditioning_model_id"),
            kwargs.get("conditioning_revision"),
        )
        auto_offload = bool(kwargs.get("auto_offload", True))
        recipe_offload = execution_recipe.get("offload_mode")
        if recipe_offload is not None:
            auto_offload = str(recipe_offload) != "none"
        offload_mode = normalize_offload_mode(
            recipe_offload if recipe_offload is not None else kwargs.get("offload_mode") or OFFLOAD_MODE_MODEL_CPU,
            auto_offload=auto_offload,
            device=device,
        )
        quantization_mode = str(kwargs.get("quantization_mode") or "none")
        quantized_components = normalize_component_list(kwargs.get("quantized_components"))
        if model_id == QWEN_IMAGE_2512_PREQUANTIZED_REPO:
            quantization_mode = "none"
            quantized_components = []
        quant_config = coerce_pipeline_quantization_config(execution_recipe.get("quantization_config"))
        if quant_config is None:
            if pipeline_class_name.startswith("QwenImage") and quantization_mode == "bnb_4bit":
                quant_config = build_qwen_pipeline_quantization_config(
                    components=quantized_components,
                    quantization_mode=quantization_mode,
                    compute_dtype=dtype,
                )
            else:
                quant_config = build_pipeline_quantization_config(quantization_mode, quantized_components, dtype)
        if quant_config is not None:
            assert_runtime_quantization_full_residency(
                model_id=model_id,
                revision=revision,
                quantization_config=quant_config,
                device=str(device),
                offload_mode=offload_mode,
                device_map=execution_recipe.get("device_map") or "none",
            )

        load_kwargs = {
            "torch_dtype": dtype,
            "revision": revision,
            "low_cpu_mem_usage": bool(kwargs.get("low_cpu_mem_usage", True)),
            "local_files_only": local_files_only(model_id),
        }
        if adapter.safe_serialization_required:
            load_kwargs["use_safetensors"] = True
        if adapter.weight_variant is not None:
            load_kwargs["variant"] = adapter.weight_variant
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
        recipe_device_map = execution_recipe.get("device_map")
        direct_device_map = kwargs.get("device_map")
        device_map = (
            direct_device_map
            if direct_device_map not in (None, "", "none")
            else recipe_device_map or direct_device_map or "none"
        )
        if device_map == "none" and offload_mode == "none" and str(device) in {"cuda", "cuda:0"}:
            # A no-offload complete pipeline is meant to be fully resident.
            # Stream its shards directly to the target accelerator instead of
            # materializing a second full CPU copy before pipeline.to(device).
            device_map = "cuda"
        if device_map == "cuda" and offload_mode == "none" and str(device) in {"cuda", "cuda:0"}:
            enable_parallel_weight_loading()
        if device_map != "none":
            load_kwargs["device_map"] = device_map
        elif quant_config is not None and str(device).startswith("cuda"):
            load_kwargs["device_map"] = "cuda"
        max_memory = execution_recipe.get("max_memory")
        if isinstance(max_memory, dict) and max_memory:
            load_kwargs["max_memory"] = max_memory

        self.progress(-1, phase="loading", message=f"Loading {pipeline_class_name}")
        if pipeline_class_name == "FluxReduxPipeline":
            from diffusers import FluxPipeline, FluxPriorReduxPipeline

            base_kwargs = dict(load_kwargs)
            # The graph revision belongs to the Redux prior. Its fixed FLUX.1
            # base is a separate Hub snapshot and must carry its own pin.
            base_kwargs["revision"] = require_catalog_revision(FLUX_DEV_REPO, model_type="FluxDevPipeline")
            with self.diffusers_loading_progress():
                base = FluxPipeline.from_pretrained(FLUX_DEV_REPO, **base_kwargs)

                prior_kwargs = dict(load_kwargs)
                prior_kwargs.pop("quantization_config", None)
                prior_kwargs.pop("device_map", None)
                # Redux exposes optional text components. Share the base FLUX
                # encoders/tokenizers so its documented prompt input is real rather
                # than silently ignored, then leave the base pipeline embedding-only.
                for component in ("text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"):
                    prior_kwargs[component] = getattr(base, component, None)
                prior = FluxPriorReduxPipeline.from_pretrained(model_id, **prior_kwargs)
            if hasattr(base, "register_modules"):
                base.register_modules(
                    text_encoder=None,
                    text_encoder_2=None,
                    tokenizer=None,
                    tokenizer_2=None,
                )
            pipeline = FluxReduxPipelineBundle(prior, base)
        elif adapter.conditioning_kind is not None:
            conditioning_model_id = repo_value(conditioning_selection)
            component_class = pipeline_class_from_name(str(adapter.conditioning_component_class))
            pipeline_class = pipeline_class_from_name(adapter.load_pipeline_class)
            component_kwargs = {
                "torch_dtype": dtype,
                "revision": conditioning_revision,
                "low_cpu_mem_usage": bool(kwargs.get("low_cpu_mem_usage", True)),
                "local_files_only": local_files_only(conditioning_model_id),
                "use_safetensors": True,
            }
            if adapter.conditioning_weight_variant is not None:
                component_kwargs["variant"] = adapter.conditioning_weight_variant
            base_load_kwargs = {
                **load_kwargs,
                "use_safetensors": True,
                str(adapter.conditioning_component_parameter): None,
            }
            with self.diffusers_loading_progress():
                conditioning_component = component_class.from_pretrained(
                    conditioning_model_id,
                    **component_kwargs,
                )
                base_load_kwargs[str(adapter.conditioning_component_parameter)] = conditioning_component
                pipeline = pipeline_class.from_pretrained(model_id, **base_load_kwargs)
        else:
            pipeline_class = pipeline_class_from_name(adapter.load_pipeline_class)
            with self.diffusers_loading_progress():
                pipeline = pipeline_class.from_pretrained(model_id, **load_kwargs)
        for component_name, component_dtype in adapter.component_dtype_overrides:
            component = getattr(pipeline, component_name, None)
            if component is None or not callable(getattr(component, "to", None)):
                raise RuntimeError(
                    f"{pipeline_class_name} did not expose its reviewed {component_name} component for dtype placement."
                )
            component.to(str_to_dtype(component_dtype))
        _tag_image_pipeline(
            pipeline,
            adapter,
            requested_mode,
            model_id,
            model_source,
            revision,
            conditioning_repo=(repo_value(conditioning_selection) if conditioning_selection is not None else None),
            conditioning_revision=conditioning_revision,
        )
        runtime_recipe = {
            **execution_recipe,
            "vae_slicing": bool(execution_recipe.get("vae_slicing", kwargs.get("enable_vae_slicing", True))),
            "vae_tiling": bool(execution_recipe.get("vae_tiling", kwargs.get("enable_vae_tiling", True))),
        }
        runtime_owner = pipeline.base if isinstance(pipeline, FluxReduxPipelineBundle) else pipeline
        pipeline._modiff_runtime_config = apply_execution_recipe_to_pipeline(runtime_owner, runtime_recipe)

        self.progress(99, phase="component_placement", message=f"Applying {offload_mode} offload")
        offload_targets = (
            [pipeline.prior, pipeline.base] if isinstance(pipeline, FluxReduxPipelineBundle) else [pipeline]
        )
        for index, target in enumerate(offload_targets):
            offload_result = apply_pipeline_offload(
                target,
                mode=offload_mode,
                device=device,
                node_id=self.node_id,
                scope=f"diffusers-image-{index}" if len(offload_targets) > 1 else "diffusers-image",
            )
            self.progress(
                -1,
                phase="component_placement",
                message=f"Registering pipeline memory ({getattr(offload_result, 'method', 'configured')})",
            )
            self.mm_add(target, priority=2)
        return {"pipeline": pipeline, "resolved_artifact": model_id}


class UnconditionalGenerate(NodeBase):
    """Generate images without prompts through a generic unconditional Diffusers pipeline."""

    label = "Diffusers Unconditional Image Generate"
    category = "Diffusers Image"
    resizable = True
    params = {
        "pipeline": {
            "label": "Pipeline",
            "display": "input",
            "type": "image_diffusion_pipeline",
            "required": True,
            "onSignal": [
                {"action": "value", "target": "image_contract"},
                {"action": "exec", "data": "update_image_contract"},
            ],
        },
        "image_contract": {
            "label": "Image Contract",
            "type": "object",
            "default": DEFAULT_UNCONDITIONAL_IMAGE_PIPELINE_CONTRACT,
            "hidden": True,
        },
        "batch_size": {"label": "Batch size", "type": "int", "default": 1, "min": 1, "max": 16},
        "seed": {"label": "Seed", "type": "int", "display": "random", "default": 0, "min": 0, "max": 4294967295},
        "num_inference_steps": {
            "label": "Steps",
            "display": "slider",
            "type": "int",
            "default": 50,
            "min": 1,
            "max": 1000,
        },
        "eta": {
            "label": "DDIM eta",
            "type": "float",
            "default": 0.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "hidden": True,
        },
        "class_label": {
            "label": "Optional class label",
            "type": "int",
            "default": -1,
            "min": -1,
            "max": 999,
            "hidden": True,
            "description": "Use -1 for unconditional sampling.",
        },
        "output_type": {"label": "Output type", "type": "string", "options": ["pil", "np"], "default": "pil"},
        "images": {"label": "Images", "display": "output", "type": "image"},
        "width_out": {"label": "Width", "display": "output", "type": "int"},
        "height_out": {"label": "Height", "display": "output", "type": "int"},
    }

    def __call__(self, **kwargs):
        _adapter, values = preflight_unconditional_action(kwargs.get("pipeline"), kwargs)
        return super().__call__(**values)

    def update_image_contract(self, values, ref):
        values = values if isinstance(values, dict) else {}
        signal_value = values.get("image_contract")
        if not isinstance(signal_value, dict):
            raise ValueError("The connected image pipeline did not publish a valid task contract.")
        adapter = get_image_pipeline_adapter(signal_value.get("pipelineClass"))
        mode = str(signal_value.get("mode") or "")
        expected_signal = image_pipeline_contract(adapter, mode)
        if signal_value != expected_signal:
            raise ValueError("The connected image pipeline published a stale or mismatched task contract.")
        if mode not in expected_signal["actions"].get(self.class_name, ()):
            raise ValueError("The connected image pipeline does not support unconditional image generation.")
        for field, params in expected_signal.get("actionFieldParams", {}).items():
            if field in self.__class__.params:
                self.set_field_params(field, params)

    def execute(self, **kwargs):
        pipeline = kwargs.get("pipeline")
        adapter, values = preflight_unconditional_action(pipeline, kwargs)

        import torch

        device = getattr(pipeline, "_execution_device", None) or getattr(pipeline, "device", None) or "cpu"
        try:
            generator = torch.Generator(device=device).manual_seed(values["seed"])
        except Exception:
            generator = torch.Generator(device="cpu").manual_seed(values["seed"])
        steps = values["num_inference_steps"]
        call_kwargs = {
            "batch_size": values["batch_size"],
            "num_inference_steps": steps,
            "generator": generator,
            "output_type": values["output_type"],
            "return_dict": True,
        }
        if "eta" in adapter.unconditional_optional_fields and supports_arg(pipeline, "eta"):
            call_kwargs["eta"] = values["eta"]
        if (
            "class_label" in adapter.unconditional_optional_fields
            and values["class_label"] >= 0
            and supports_arg(pipeline, "class_labels")
        ):
            call_kwargs["class_labels"] = values["class_label"]
        add_progress_callback(self, pipeline, call_kwargs, steps)
        if supports_arg(pipeline, "callback") and "callback_on_step_end" not in call_kwargs:
            pipeline._num_timesteps = steps

            def callback(step_index, timestep, latents):
                self.pipe_callback(pipeline, step_index, timestep, {})

            call_kwargs["callback"] = callback
            if supports_arg(pipeline, "callback_steps"):
                call_kwargs["callback_steps"] = 1
        self._active_pipeline = pipeline
        try:
            result = pipeline(**call_kwargs)
        finally:
            self._active_pipeline = None
        images = getattr(result, "images", result)
        width, height = output_image_dimensions(images, values["output_type"])
        return {"images": images, "width_out": width, "height_out": height}


class PredictMap(NodeBase):
    """Predict a normalized semantic map with a compatible Diffusers perception pipeline."""

    label = "Diffusers Predict Map"
    category = "Diffusers Image"
    resizable = True
    params = {
        "pipeline": {
            "label": "Pipeline",
            "display": "input",
            "type": "image_diffusion_pipeline",
            "required": True,
            "onSignal": [
                {"action": "value", "target": "image_contract"},
                {"action": "exec", "data": "update_image_contract"},
            ],
        },
        "image_contract": {
            "label": "Image Contract",
            "type": "object",
            "default": DEFAULT_PREDICTION_MAP_PIPELINE_CONTRACT,
            "hidden": True,
        },
        "image": {"label": "Source image", "display": "input", "type": "image", "required": True},
        "prediction_kind": {
            "label": "Map kind",
            "type": "string",
            "options": ["depth"],
            "default": "depth",
        },
        "seed": {
            "label": "Seed",
            "type": "int",
            "display": "random",
            "default": 0,
            "min": 0,
            "max": 4294967295,
        },
        "num_inference_steps": {
            "label": "Steps",
            "display": "slider",
            "type": "int",
            "default": 1,
            "min": 1,
            "max": 50,
        },
        "processing_resolution": {
            "label": "Processing resolution",
            "type": "int",
            "default": 768,
            "min": 64,
            "max": 2048,
            "step": 8,
            "description": "Longest edge used internally by the perception model.",
        },
        "match_input_resolution": {
            "label": "Match input resolution",
            "type": "bool",
            "default": True,
        },
        "prediction_map": {"label": "Prediction map", "display": "output", "type": "prediction_map"},
        "preview_images": {"label": "Preview", "display": "output", "type": "image"},
        "width_out": {"label": "Width", "display": "output", "type": "int"},
        "height_out": {"label": "Height", "display": "output", "type": "int"},
    }

    def __call__(self, **kwargs):
        _adapter, values = preflight_prediction_map_action(kwargs.get("pipeline"), kwargs)
        return super().__call__(**values)

    def update_image_contract(self, values, ref):
        values = values if isinstance(values, dict) else {}
        signal_value = values.get("image_contract")
        if not isinstance(signal_value, dict):
            raise ValueError("The connected image pipeline did not publish a valid task contract.")
        adapter = get_image_pipeline_adapter(signal_value.get("pipelineClass"))
        mode = str(signal_value.get("mode") or "")
        expected_signal = image_pipeline_contract(adapter, mode)
        if signal_value != expected_signal:
            raise ValueError("The connected image pipeline published a stale or mismatched task contract.")
        if mode not in expected_signal["actions"].get(self.class_name, ()):
            raise ValueError("The connected image pipeline does not support generic prediction maps.")

    def execute(self, **kwargs):
        pipeline = kwargs.get("pipeline")
        _adapter, values = preflight_prediction_map_action(pipeline, kwargs)

        import torch

        device = getattr(pipeline, "_execution_device", None) or getattr(pipeline, "device", None) or "cpu"
        try:
            generator = torch.Generator(device=device).manual_seed(values["seed"])
        except Exception:
            generator = torch.Generator(device="cpu").manual_seed(values["seed"])
        call_kwargs = {
            "image": values["image"],
            "num_inference_steps": values["num_inference_steps"],
            "ensemble_size": 1,
            "processing_resolution": values["processing_resolution"],
            "match_input_resolution": values["match_input_resolution"],
            "generator": generator,
            "output_type": "np",
            "output_uncertainty": False,
            "output_latent": False,
            "return_dict": True,
        }
        self._active_pipeline = pipeline
        try:
            result = pipeline(**call_kwargs)
        finally:
            self._active_pipeline = None
        prediction = result.get("prediction") if isinstance(result, dict) else getattr(result, "prediction", None)
        if prediction is None:
            raise ValueError("Diffusers perception pipeline did not return a prediction map.")
        prediction_map, preview_images = normalize_prediction_map(
            prediction,
            kind=values["prediction_kind"],
        )
        return {
            "prediction_map": prediction_map,
            "preview_images": preview_images,
            "width_out": prediction_map["width"],
            "height_out": prediction_map["height"],
        }


class Generate(NodeBase):
    """Generate images from text with a Diffusers image pipeline."""

    label = "Diffusers Image Generate"
    category = "Diffusers Image"
    resizable = True
    params = {
        "pipeline": {
            "label": "Pipeline",
            "display": "input",
            "type": "image_diffusion_pipeline",
            "required": True,
            "onSignal": [
                {"action": "value", "target": "image_contract"},
                {"action": "exec", "data": "update_image_contract"},
            ],
        },
        "image_contract": {
            "label": "Image Contract",
            "type": "object",
            "default": DEFAULT_IMAGE_PIPELINE_CONTRACT,
            "hidden": True,
        },
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "negative_prompt": {"label": "Negative Prompt", "display": "textarea", "type": "text", "default": ""},
        "width": {"label": "Width", "type": "int", "default": 1024, "min": 16, "max": 2048, "step": 16},
        "height": {"label": "Height", "type": "int", "default": 1024, "min": 16, "max": 2048, "step": 16},
        "seed": {"label": "Seed", "type": "int", "display": "random", "default": 0, "min": 0, "max": 4294967295},
        "num_inference_steps": {
            "label": "Steps",
            "display": "slider",
            "type": "int",
            "default": 4,
            "min": 1,
            "max": 100,
        },
        "guidance_scale": {
            "label": "Guidance",
            "display": "slider",
            "type": "float",
            "default": 0.0,
            "min": 0,
            "max": 20,
            "step": 0.1,
        },
        "image_guidance_scale": {
            "label": "Image Guidance",
            "display": "slider",
            "type": "float",
            "default": 1.5,
            "min": 1,
            "max": 20,
            "step": 0.1,
            "hidden": True,
        },
        "pag_scale": {
            "label": "PAG Scale",
            "display": "slider",
            "type": "float",
            "default": 3.0,
            "min": 0,
            "max": 20,
            "step": 0.1,
            "hidden": True,
        },
        "pag_adaptive_scale": {
            "label": "PAG Adaptive Scale",
            "display": "slider",
            "type": "float",
            "default": 0.0,
            "min": 0,
            "max": 20,
            "step": 0.1,
            "hidden": True,
        },
        "strength": {
            "label": "Strength",
            "display": "slider",
            "type": "float",
            "default": 0.8,
            "min": 0,
            "max": 1,
            "step": 0.01,
            "hidden": True,
        },
        "padding_mask_crop": {
            "label": "Padding Mask Crop",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 512,
            "step": 8,
            "hidden": True,
        },
        "max_sequence_length": {"label": "Max Sequence Length", "type": "int", "default": 256, "min": 1, "max": 512},
        "output_type": {"label": "Output type", "type": "string", "options": ["pil", "np", "pt"], "default": "pil"},
        "images": {"label": "Images", "display": "output", "type": "image"},
        "width_out": {"label": "Width", "display": "output", "type": "int"},
        "height_out": {"label": "Height", "display": "output", "type": "int"},
    }

    def __call__(self, **kwargs):
        # Validate the raw graph payload before NodeBase can coerce booleans,
        # blank strings, or container values into apparently valid numbers.
        # The concrete registered class owns the action/mode check, so this
        # remains one generic facade for Generate/Edit/Inpaint/ControlGenerate.
        action = self.class_name
        if action not in {"Generate", "Edit", "Inpaint", "ControlGenerate"}:
            raise ValueError(f"Unsupported Diffusers image action {action!r}.")
        _adapter, values = preflight_image_action(kwargs.get("pipeline"), action, kwargs)
        return super().__call__(**values)

    def update_image_contract(self, values, ref):
        """Apply the loader's backend-owned contract to this generic image form."""

        values = values if isinstance(values, dict) else {}
        signal_value = values.get("image_contract")
        if not isinstance(signal_value, dict):
            raise ValueError("The connected image pipeline did not publish a valid task contract.")
        adapter = get_image_pipeline_adapter(signal_value.get("pipelineClass"))
        mode = str(signal_value.get("mode") or "")
        expected_signal = image_pipeline_contract(adapter, mode)
        if signal_value != expected_signal:
            raise ValueError("The connected image pipeline published a stale or mismatched task contract.")
        if mode not in expected_signal["actions"].get(self.class_name, ()):
            raise ValueError("The connected image pipeline does not support this generic image action.")

        for field, params in expected_signal["fieldParams"].items():
            if field in self.__class__.params:
                self.set_field_params(field, params)

    def execute(self, **kwargs):
        pipeline = kwargs.get("pipeline")
        adapter, values = preflight_image_action(pipeline, "Generate", kwargs)

        import torch

        device = getattr(pipeline, "_execution_device", None) or getattr(pipeline, "device", None) or "cpu"
        try:
            generator = torch.Generator(device=device).manual_seed(values["seed"])
        except Exception:
            generator = torch.Generator(device="cpu").manual_seed(values["seed"])
        steps = values["num_inference_steps"]
        call_kwargs = {
            "prompt": values.get("prompt") or "",
            "width": values["width"],
            "height": values["height"],
            "num_inference_steps": steps,
            "generator": generator,
            "output_type": values["output_type"],
            "return_dict": True,
        }
        adapter.apply_generation_parameters(pipeline, values, call_kwargs)
        add_progress_callback(self, pipeline, call_kwargs, steps)
        self._active_pipeline = pipeline
        try:
            result = pipeline(**call_kwargs)
        finally:
            self._active_pipeline = None
        images = getattr(result, "images", result)
        return {"images": images, "width_out": call_kwargs["width"], "height_out": call_kwargs["height"]}


class Edit(Generate):
    """Edit an image with a Diffusers image pipeline."""

    label = "Diffusers Image Edit"
    category = "Diffusers Image"
    params = {
        **Generate.params,
        "image": {
            "label": "Image or references",
            "display": "input",
            "type": "image",
            "required": True,
            "description": "A single source image or a list of references for pipelines that support multi-reference editing.",
        },
        "reference_strength": {
            "label": "Secondary reference strength",
            "type": "float",
            "default": 1.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "hidden": True,
            "description": "Relative influence of every reference after the first composition anchor, when supported by the selected adapter.",
        },
    }

    def execute(self, **kwargs):
        pipeline = kwargs.get("pipeline")
        adapter, values = preflight_image_action(pipeline, "Edit", kwargs)
        image = prepare_reference_images(values.get("image"), adapter)
        return self._execute_conditioned(values, {"image": image}, adapter=adapter)

    def _execute_conditioned(self, values, extra_kwargs, *, adapter):
        pipeline = values["pipeline"]

        import torch

        device = getattr(pipeline, "_execution_device", None) or getattr(pipeline, "device", None) or "cpu"
        try:
            generator = torch.Generator(device=device).manual_seed(values["seed"])
        except Exception:
            generator = torch.Generator(device="cpu").manual_seed(values["seed"])
        steps = values["num_inference_steps"]
        call_kwargs = {
            "prompt": values.get("prompt") or "",
            "num_inference_steps": steps,
            "generator": generator,
            "output_type": values["output_type"],
            "return_dict": True,
            **extra_kwargs,
        }
        adapter.apply_generation_parameters(pipeline, values, call_kwargs)
        add_progress_callback(self, pipeline, call_kwargs, steps)
        self._active_pipeline = pipeline
        try:
            result = pipeline(**call_kwargs)
        finally:
            self._active_pipeline = None
        images = getattr(result, "images", result)
        actual_width, actual_height = output_image_dimensions(images, call_kwargs["output_type"])
        return {
            "images": images,
            "width_out": actual_width if actual_width is not None else values["width"],
            "height_out": actual_height if actual_height is not None else values["height"],
        }


class Inpaint(Edit):
    """Inpaint or fill with a Diffusers image pipeline."""

    label = "Diffusers Image Inpaint"
    category = "Diffusers Image"
    params = {
        **Edit.params,
        "mask_image": {"label": "Mask", "display": "input", "type": "image", "required": True},
        "output_type": {"label": "Output type", "type": "string", "options": ["pil"], "default": "pil"},
    }

    def execute(self, **kwargs):
        pipeline = kwargs.get("pipeline")
        adapter, values = preflight_image_action(pipeline, "Inpaint", kwargs)
        result = self._execute_conditioned(
            values,
            {"image": values["image"], "mask_image": values["mask_image"]},
            adapter=adapter,
        )
        result["images"] = composite_masked_pil_outputs(
            result.get("images"),
            values["image"],
            values["mask_image"],
        )
        return result


class ControlGenerate(Edit):
    """Generate from a control image with a Diffusers image pipeline."""

    label = "Diffusers Control Generate"
    category = "Diffusers Image"
    params = {
        **Generate.params,
        "control_image": {"label": "Control Image", "display": "input", "type": "image", "required": True},
        "conditioning_scale": {
            "label": "Conditioning Scale",
            "display": "slider",
            "type": "float",
            "default": 1.0,
            "min": 0.0,
            "max": 2.0,
            "step": 0.05,
            "hidden": True,
        },
    }

    def execute(self, **kwargs):
        pipeline = kwargs.get("pipeline")
        adapter, values = preflight_image_action(pipeline, "ControlGenerate", kwargs)
        return self._execute_conditioned(
            values,
            {adapter.control_image_parameter: values["control_image"]},
            adapter=adapter,
        )


class LoadAdapter(NodeBase):
    """Load a LoRA or adapter into a Diffusers image pipeline."""

    label = "Load Diffusers Image Adapter"
    category = "Diffusers Image"
    resizable = True
    params = {
        "pipeline": {"label": "Pipeline", "display": "input", "type": "image_diffusion_pipeline", "required": True},
        "adapter_path": {
            "label": "Adapter",
            "display": "modelselect",
            "type": "string",
            "fieldOptions": {"noValidation": True, "sources": ["hub", "local"]},
        },
        "weight_name": {"label": "Weight name", "type": "string", "default": ""},
        "revision": {
            "label": "Revision",
            "type": "string",
            "default": "",
            "description": "Required immutable Hub commit for the selected adapter repository.",
        },
        "expected_sha256": {
            "label": "Expected SHA-256",
            "type": "string",
            "default": "",
            "description": "Required SHA-256 for Hub adapter weights; optional for local weights.",
        },
        "adapter_name": {"label": "Adapter name", "type": "string", "default": "default"},
        "replace_existing": {
            "label": "Replace existing adapters",
            "type": "boolean",
            "default": True,
            "description": "Unload adapters already attached to this pipeline before loading this graph's adapter.",
        },
        "scale": {
            "label": "Scale",
            "display": "slider",
            "type": "float",
            "default": 1.0,
            "min": -2,
            "max": 2,
            "step": 0.01,
        },
        "output": {"label": "Pipeline", "display": "output", "type": "image_diffusion_pipeline"},
    }

    @staticmethod
    def _normalized_request(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Normalize the raw adapter contract without touching a pipeline."""

        if not isinstance(kwargs, dict):
            raise ValueError("Diffusers image adapter values must be an object.")
        pipeline = kwargs.get("pipeline")
        if pipeline is None:
            raise ValueError("LoadAdapter needs a pipeline input.")
        if not callable(getattr(pipeline, "load_lora_weights", None)):
            raise ValueError("This pipeline does not expose load_lora_weights().")

        selection = kwargs.get("adapter_path")
        if selection is None or (isinstance(selection, str) and not selection.strip()):
            source = None
            adapter_path = ""
            normalized_selection: str | dict[str, str] = ""
        elif isinstance(selection, dict):
            source = _canonical_image_model_source(selection.get("source"), label="Diffusers image adapter")
            raw_adapter_path = selection.get("value")
            if not isinstance(raw_adapter_path, str):
                raise ValueError("Diffusers image adapter value must be a repository ID or local path string.")
            adapter_path = raw_adapter_path.strip()
            if not adapter_path:
                source = None
                normalized_selection = ""
            else:
                normalized_selection = {"source": source, "value": adapter_path}
        elif isinstance(selection, str):
            source = "hub"
            adapter_path = selection.strip()
            normalized_selection = {"source": source, "value": adapter_path}
        else:
            raise ValueError(
                "Diffusers image adapter selection must be a repository ID or a hub/local selection object."
            )

        raw_weight_name = kwargs.get("weight_name")
        if raw_weight_name is not None and not isinstance(raw_weight_name, str):
            raise ValueError("Diffusers image adapter weight_name must be a string.")
        weight_name = str(raw_weight_name or "").strip()

        raw_revision = kwargs.get("revision")
        if raw_revision is not None and not isinstance(raw_revision, str):
            raise ValueError("Diffusers image adapter revision must be a string.")
        revision = str(raw_revision or "").strip()
        if isinstance(raw_revision, str) and raw_revision != revision:
            raise ValueError("Diffusers image adapter revision must be an exact trimmed string.")

        raw_expected_sha256 = kwargs.get("expected_sha256")
        if raw_expected_sha256 is not None and not isinstance(raw_expected_sha256, str):
            raise ValueError("Diffusers image adapter expected_sha256 must be a string.")
        expected_sha256 = str(raw_expected_sha256 or "").strip().lower().removeprefix("sha256:")
        if expected_sha256 and (
            len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError("Diffusers image adapter expected_sha256 must contain exactly 64 hexadecimal digits.")

        raw_adapter_name = kwargs.get("adapter_name")
        if raw_adapter_name is not None and not isinstance(raw_adapter_name, str):
            raise ValueError("Diffusers image adapter name must be a string.")
        adapter_name = str(raw_adapter_name or "default").strip()
        if not adapter_name:
            raise ValueError("Diffusers image adapter name cannot be blank.")
        scale = _bounded_image_float(
            kwargs.get("scale"), field="adapter scale", default=1.0, minimum=-2.0, maximum=2.0
        )
        raw_replace_existing = kwargs.get("replace_existing")
        if raw_replace_existing is None:
            replace_existing = True
        elif type(raw_replace_existing) is not bool:
            raise ValueError("Diffusers image adapter replace_existing must be a boolean.")
        else:
            replace_existing = raw_replace_existing

        if source == "hub":
            adapter_path = _validated_image_hub_repository(adapter_path, label="Diffusers image adapter repository")
            normalized_selection = {"source": "hub", "value": adapter_path}
            if not weight_name:
                raise ValueError("A Hub adapter requires an exact safetensors weight_name.")
            weight_parts = weight_name.split("/")
            if (
                "\\" in weight_name
                or weight_name.startswith("/")
                or any(part in {"", ".", ".."} for part in weight_parts)
                or not weight_name.endswith(".safetensors")
            ):
                raise ValueError("A Hub adapter weight_name must be a contained .safetensors repository file.")
            if revision != revision.lower() or not revision or not IMMUTABLE_HUB_REVISION.fullmatch(revision):
                raise ValueError("A Hub Diffusers image adapter requires a lowercase 40-character commit revision.")
            if not expected_sha256:
                raise ValueError("A Hub Diffusers image adapter requires an expected SHA-256 hash.")
        elif source == "local":
            if revision:
                raise ValueError("A local Diffusers image adapter cannot carry a Hub revision.")
        elif revision or expected_sha256 or weight_name:
            raise ValueError("Adapter weight, revision, and hash values require a selected adapter.")

        values = dict(kwargs)
        values.update(
            {
                "pipeline": pipeline,
                "adapter_path": normalized_selection,
                "weight_name": weight_name,
                "revision": revision,
                "expected_sha256": expected_sha256,
                "adapter_name": adapter_name,
                "replace_existing": replace_existing,
                "scale": scale,
            }
        )
        return values

    def __call__(self, **kwargs):
        # This facade-local pass prevents NodeBase's permissive primitive casts
        # from turning malformed security fields into another request.
        return super().__call__(**self._normalized_request(kwargs))

    def execute(self, **kwargs):
        values = self._normalized_request(kwargs)
        pipeline = values["pipeline"]
        selection = values["adapter_path"]
        if not selection:
            return {"output": pipeline}
        source = selection["source"]
        adapter_path = selection["value"]
        weight_name = values["weight_name"] or None
        resolved_weight: Path
        load_adapter_path: Path
        load_weight_name: str
        if source == "hub":
            repo_id = adapter_path
            cached = cached_file_path(repo_id, weight_name, revision=values["revision"])
            if not cached:
                raise FileNotFoundError(
                    f"Adapter {repo_id}/{weight_name} is not installed. Install the pinned file through Model Manager first."
                )
            cached_alias = Path(cached).expanduser()
            if not cached_alias.name.endswith(".safetensors"):
                raise ValueError(
                    "The installed Hub adapter snapshot entry must have a lowercase .safetensors filename."
                )
            try:
                resolved_weight = resolve_managed_hf_cache_file(cached)
            except (FileNotFoundError, ValueError) as error:
                raise FileNotFoundError(
                    f"Installed adapter cache entry does not exist: {repo_id}/{weight_name}. "
                    "Repair it through Model Manager."
                ) from error
            # Hugging Face snapshot entries normally symlink to extensionless
            # blob files. Hash the resolved blob, but retain the validated
            # snapshot alias so Diffusers selects its safetensors-only branch.
            load_adapter_path = cached_alias.parent
            load_weight_name = cached_alias.name
        else:
            try:
                local_target = Path(adapter_path).expanduser().resolve(strict=True)
            except (OSError, RuntimeError) as error:
                raise FileNotFoundError(f"Diffusers image adapter path does not exist: {adapter_path}") from error
            if local_target.is_file():
                if weight_name is not None and Path(weight_name).name != local_target.name:
                    raise ValueError("Local adapter file selection and weight_name refer to different files.")
                resolved_weight = local_target
            elif local_target.is_dir():
                if weight_name is None:
                    raise ValueError("A local Diffusers image adapter directory requires an exact weight_name.")
                requested_weight = Path(weight_name)
                if requested_weight.is_absolute():
                    raise ValueError("Local Diffusers image adapter weight_name must stay inside its selected folder.")
                try:
                    resolved_weight = (local_target / requested_weight).resolve(strict=True)
                    resolved_weight.relative_to(local_target)
                except (OSError, RuntimeError, ValueError) as error:
                    raise FileNotFoundError(
                        f"Local Diffusers image adapter weight does not exist inside the selected folder: {weight_name}"
                    ) from error
                if not resolved_weight.is_file():
                    raise FileNotFoundError("The selected local Diffusers image adapter weight is not a file.")
            else:
                raise FileNotFoundError(f"Diffusers image adapter target is not a file or folder: {local_target}")

            if not resolved_weight.name.endswith(".safetensors"):
                raise ValueError("A local Diffusers image adapter must select a lowercase .safetensors file.")
            load_adapter_path = resolved_weight.parent
            load_weight_name = resolved_weight.name

        expected_sha256 = values["expected_sha256"]
        if expected_sha256:
            digest = hashlib.sha256()
            with resolved_weight.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise ValueError(
                    "The Diffusers image adapter failed its pinned SHA-256 verification. "
                    "Repair or reselect it before running this graph."
                )

        adapter_name = values["adapter_name"]
        scale = values["scale"]
        load_kwargs = {
            "adapter_name": adapter_name,
            "weight_name": load_weight_name,
            "use_safetensors": True,
        }
        adapter_path = str(load_adapter_path)
        replace_existing = values["replace_existing"]
        adapter_scales = dict(getattr(pipeline, "_modiff_adapter_scales", {}) or {})
        unload_lora_weights = getattr(pipeline, "unload_lora_weights", None)
        if replace_existing and callable(unload_lora_weights):
            unload_lora_weights()
            adapter_scales.clear()
        pipeline.load_lora_weights(adapter_path, **load_kwargs)
        adapter_scales[adapter_name] = scale
        if callable(getattr(pipeline, "set_adapters", None)):
            pipeline.set_adapters(list(adapter_scales), list(adapter_scales.values()))
        pipeline._modiff_adapter_scales = adapter_scales
        return {"output": pipeline}
