from __future__ import annotations

from dataclasses import asdict, dataclass

from modiff.diffusers_offload_modes import (
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
)


QWEN_IMAGE_2512_REPO = "Qwen/Qwen-Image-2512"
QWEN_IMAGE_2512_PREQUANTIZED_REPO = "unsloth/Qwen-Image-2512-unsloth-bnb-4bit"
ACE_STEP_REPO = "ACE-Step/acestep-v15-xl-turbo-diffusers"
FLUX_SCHNELL_REPO = "black-forest-labs/FLUX.1-schnell"
FLUX_DEV_REPO = "black-forest-labs/FLUX.1-dev"
FLUX_KREA_REPO = "black-forest-labs/FLUX.1-Krea-dev"
FLUX_KONTEXT_REPO = "black-forest-labs/FLUX.1-Kontext-dev"
FLUX_FILL_REPO = "black-forest-labs/FLUX.1-Fill-dev"
FLUX_DEPTH_REPO = "black-forest-labs/FLUX.1-Depth-dev"
FLUX_CANNY_REPO = "black-forest-labs/FLUX.1-Canny-dev"
FLUX_CANNY_VERIFIED_REPAIR_REPO = "fuliucansheng/FLUX.1-Canny-dev-diffusers"
FLUX_REDUX_REPO = "black-forest-labs/FLUX.1-Redux-dev"
FLUX2_KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
LTX_VIDEO_REPO = "Lightricks/LTX-Video-0.9.8-13B-distilled"
LTX_VIDEO_FALLBACK_REPO = "Lightricks/LTX-Video"
WAN_T2V_1_3B_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN_22_TI2V_5B_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

VERIFIED_REPAIR_SOURCES = {
    FLUX_CANNY_REPO: FLUX_CANNY_VERIFIED_REPAIR_REPO,
}


@dataclass(frozen=True)
class DiffusersExecutionProfile:
    id: str
    model_type: str
    modes: tuple[str, ...]
    backend_path: str
    pipeline_class: str
    default_repo: str
    fallback_repo: str | None
    quantizable_components: tuple[str, ...]
    default_quantized_components: tuple[str, ...]
    supported_offload_modes: tuple[str, ...]
    retry_offload_modes: tuple[str, ...]
    max_low_memory_side: int | None
    max_low_memory_steps: int | None
    live_proof: bool

    def to_public_dict(self) -> dict:
        data = asdict(self)
        return {key: list(value) if isinstance(value, tuple) else value for key, value in data.items()}


DIFFUSERS_EXECUTION_PROFILES: dict[str, DiffusersExecutionProfile] = {
    "z-image:auto": DiffusersExecutionProfile(
        id="z-image:auto",
        model_type="ZImageModularPipeline",
        modes=("text_to_image",),
        backend_path="modules.ModularDiffusers.ModelsLoader",
        pipeline_class="ZImageModularPipeline",
        default_repo="Tongyi-MAI/Z-Image-Turbo",
        fallback_repo=None,
        quantizable_components=(),
        default_quantized_components=(),
        supported_offload_modes=(OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
        retry_offload_modes=(OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK),
        max_low_memory_side=1024,
        max_low_memory_steps=8,
        live_proof=False,
    ),
    "qwen-image:t2i-direct": DiffusersExecutionProfile(
        id="qwen-image:t2i-direct",
        model_type="QwenImageModularPipeline",
        modes=("text_to_image",),
        backend_path="modules.DiffusersImage.LoadPipeline",
        pipeline_class="QwenImagePipeline",
        default_repo=QWEN_IMAGE_2512_REPO,
        fallback_repo=QWEN_IMAGE_2512_PREQUANTIZED_REPO,
        quantizable_components=("transformer", "text_encoder"),
        default_quantized_components=(),
        supported_offload_modes=(
            OFFLOAD_MODE_NONE,
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        ),
        retry_offload_modes=(OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK),
        max_low_memory_side=1328,
        max_low_memory_steps=50,
        live_proof=False,
    ),
    "qwen-image:modular": DiffusersExecutionProfile(
        id="qwen-image:modular",
        model_type="QwenImageModularPipeline",
        modes=("control_image",),
        backend_path="modules.ModularDiffusers.ModelsLoader",
        pipeline_class="QwenImageModularPipeline",
        default_repo=QWEN_IMAGE_2512_REPO,
        fallback_repo=None,
        quantizable_components=("transformer", "text_encoder"),
        default_quantized_components=("transformer", "text_encoder"),
        supported_offload_modes=(OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
        retry_offload_modes=(OFFLOAD_MODE_GROUP_DISK,),
        max_low_memory_side=768,
        max_low_memory_steps=28,
        live_proof=False,
    ),
    "qwen-edit:direct-inpaint": DiffusersExecutionProfile(
        id="qwen-edit:direct-inpaint",
        model_type="QwenImageEditModularPipeline",
        modes=("inpaint", "outpaint"),
        backend_path="modules.DiffusersImage.LoadPipeline",
        pipeline_class="QwenImageEditInpaintPipeline",
        default_repo="Qwen/Qwen-Image-Edit",
        fallback_repo=None,
        quantizable_components=("transformer", "text_encoder"),
        default_quantized_components=("transformer", "text_encoder"),
        supported_offload_modes=(
            OFFLOAD_MODE_NONE,
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        ),
        retry_offload_modes=(OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK),
        max_low_memory_side=768,
        max_low_memory_steps=24,
        live_proof=False,
    ),
    "qwen-edit:modular": DiffusersExecutionProfile(
        id="qwen-edit:modular",
        model_type="QwenImageEditModularPipeline",
        modes=("edit_image",),
        backend_path="modules.ModularDiffusers.ModelsLoader",
        pipeline_class="QwenImageEditModularPipeline",
        default_repo="Qwen/Qwen-Image-Edit",
        fallback_repo=None,
        quantizable_components=("transformer", "text_encoder"),
        default_quantized_components=("transformer", "text_encoder"),
        supported_offload_modes=(OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
        retry_offload_modes=(OFFLOAD_MODE_GROUP_DISK,),
        max_low_memory_side=768,
        max_low_memory_steps=24,
        live_proof=False,
    ),
    "qwen-edit-plus:modular": DiffusersExecutionProfile(
        id="qwen-edit-plus:modular",
        model_type="QwenImageEditPlusModularPipeline",
        modes=("edit_image", "multi_image_reference_edit"),
        backend_path="modules.ModularDiffusers.ModelsLoader",
        pipeline_class="QwenImageEditPlusModularPipeline",
        default_repo="Qwen/Qwen-Image-Edit-2511",
        fallback_repo=None,
        quantizable_components=("transformer", "text_encoder"),
        default_quantized_components=("transformer", "text_encoder"),
        supported_offload_modes=(OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
        retry_offload_modes=(OFFLOAD_MODE_GROUP_DISK,),
        max_low_memory_side=768,
        max_low_memory_steps=24,
        live_proof=False,
    ),
    "qwen-layered:modular": DiffusersExecutionProfile(
        id="qwen-layered:modular",
        model_type="QwenImageLayeredModularPipeline",
        modes=("layer_decomposition",),
        backend_path="modules.ModularDiffusers.ModelsLoader",
        pipeline_class="QwenImageLayeredModularPipeline",
        default_repo="Qwen/Qwen-Image-Layered",
        fallback_repo=None,
        quantizable_components=("transformer", "text_encoder"),
        default_quantized_components=("transformer", "text_encoder"),
        supported_offload_modes=(OFFLOAD_MODE_NONE, OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
        retry_offload_modes=(OFFLOAD_MODE_GROUP_DISK,),
        max_low_memory_side=768,
        max_low_memory_steps=30,
        live_proof=False,
    ),
    "wan-vace:direct": DiffusersExecutionProfile(
        id="wan-vace:direct",
        model_type="WanVACEPipeline",
        modes=(
            "text_to_video",
            "video_inpaint",
            "video_outpaint",
            "control_to_video",
        ),
        backend_path="modules.DiffusersVideo.LoadPipeline",
        pipeline_class="WanVACEPipeline",
        default_repo="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        fallback_repo=None,
        quantizable_components=(),
        default_quantized_components=(),
        supported_offload_modes=(
            OFFLOAD_MODE_NONE,
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        ),
        retry_offload_modes=(OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
        max_low_memory_side=832,
        max_low_memory_steps=24,
        live_proof=False,
    ),
    "wan-video-to-video:direct": DiffusersExecutionProfile(
        id="wan-video-to-video:direct",
        model_type="WanVideoPipeline",
        modes=("video_to_video", "video_color_edit"),
        backend_path="modules.DiffusersVideo.LoadPipeline",
        pipeline_class="WanVideoToVideoPipeline",
        default_repo=WAN_T2V_1_3B_REPO,
        fallback_repo=None,
        quantizable_components=(),
        default_quantized_components=(),
        supported_offload_modes=(
            OFFLOAD_MODE_NONE,
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        ),
        retry_offload_modes=(OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK),
        max_low_memory_side=832,
        max_low_memory_steps=30,
        live_proof=False,
    ),
    "wan-text-to-video:direct": DiffusersExecutionProfile(
        id="wan-text-to-video:direct",
        model_type="WanVideoPipeline",
        modes=("text_to_video",),
        backend_path="modules.DiffusersVideo.LoadPipeline",
        pipeline_class="WanPipeline",
        default_repo=WAN_T2V_1_3B_REPO,
        fallback_repo=None,
        quantizable_components=(),
        default_quantized_components=(),
        supported_offload_modes=(
            OFFLOAD_MODE_NONE,
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        ),
        retry_offload_modes=(OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK),
        max_low_memory_side=832,
        max_low_memory_steps=30,
        live_proof=True,
    ),
    "wan-22-image-to-video:direct": DiffusersExecutionProfile(
        id="wan-22-image-to-video:direct",
        model_type="WanImageToVideoPipeline",
        modes=("image_to_video",),
        backend_path="modules.DiffusersVideo.LoadPipeline",
        pipeline_class="WanImageToVideoPipeline",
        default_repo="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        fallback_repo=None,
        quantizable_components=("transformer", "transformer_2", "text_encoder"),
        default_quantized_components=("transformer", "transformer_2"),
        supported_offload_modes=(
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        ),
        retry_offload_modes=(OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK),
        max_low_memory_side=832,
        max_low_memory_steps=40,
        live_proof=False,
    ),
    "wan-22-ti2v-5b:direct": DiffusersExecutionProfile(
        id="wan-22-ti2v-5b:direct",
        model_type="WanTI2VPipeline",
        modes=("text_to_video",),
        backend_path="modules.DiffusersVideo.LoadPipeline",
        pipeline_class="WanTI2VPipeline",
        default_repo=WAN_22_TI2V_5B_REPO,
        fallback_repo=None,
        quantizable_components=("transformer", "text_encoder"),
        default_quantized_components=(),
        supported_offload_modes=(
            OFFLOAD_MODE_NONE,
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        ),
        retry_offload_modes=(OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
        max_low_memory_side=1280,
        max_low_memory_steps=50,
        live_proof=False,
    ),
    "ltx-video:direct": DiffusersExecutionProfile(
        id="ltx-video:direct",
        model_type="LTXVideoPipeline",
        modes=("text_to_video", "image_to_video", "video_to_video", "reference_to_video"),
        backend_path="modules.DiffusersVideo.LoadPipeline",
        pipeline_class="LTXConditionPipeline",
        default_repo=LTX_VIDEO_REPO,
        fallback_repo=LTX_VIDEO_FALLBACK_REPO,
        quantizable_components=("transformer", "text_encoder"),
        default_quantized_components=(),
        supported_offload_modes=(
            OFFLOAD_MODE_NONE,
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        ),
        retry_offload_modes=(OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_GROUP_DISK),
        max_low_memory_side=704,
        max_low_memory_steps=8,
        live_proof=False,
    ),
    "ace-step-audio:direct": DiffusersExecutionProfile(
        id="ace-step-audio:direct",
        model_type="AceStepAudioPipeline",
        modes=("text_to_audio", "audio_variation", "audio_continuation", "audio_repaint"),
        backend_path="modules.DiffusersAudio.LoadPipeline",
        pipeline_class="AceStepPipeline",
        default_repo=ACE_STEP_REPO,
        fallback_repo=None,
        quantizable_components=(),
        default_quantized_components=(),
        supported_offload_modes=(
            OFFLOAD_MODE_NONE,
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        ),
        retry_offload_modes=(OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK),
        max_low_memory_side=None,
        max_low_memory_steps=8,
        live_proof=False,
    ),
    "flux-schnell:direct": DiffusersExecutionProfile(
        id="flux-schnell:direct",
        model_type="FluxSchnellPipeline",
        modes=("text_to_image",),
        backend_path="modules.DiffusersImage.LoadPipeline",
        pipeline_class="FluxPipeline",
        default_repo=FLUX_SCHNELL_REPO,
        fallback_repo=None,
        quantizable_components=("transformer", "text_encoder_2"),
        default_quantized_components=(),
        supported_offload_modes=(
            OFFLOAD_MODE_NONE,
            OFFLOAD_MODE_MODEL_CPU,
            OFFLOAD_MODE_SEQUENTIAL_CPU,
            OFFLOAD_MODE_GROUP_CPU,
            OFFLOAD_MODE_GROUP_DISK,
        ),
        retry_offload_modes=(OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK),
        max_low_memory_side=1024,
        max_low_memory_steps=4,
        live_proof=False,
    ),
    "flux-dev:direct": DiffusersExecutionProfile(
        id="flux-dev:direct",
        model_type="FluxDevPipeline",
        modes=("text_to_image",),
        backend_path="modules.DiffusersImage.LoadPipeline",
        pipeline_class="FluxPipeline",
        default_repo=FLUX_DEV_REPO,
        fallback_repo=None,
        quantizable_components=("transformer", "text_encoder_2"),
        default_quantized_components=("transformer",),
        supported_offload_modes=(OFFLOAD_MODE_MODEL_CPU, OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_CPU, OFFLOAD_MODE_GROUP_DISK),
        retry_offload_modes=(OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK),
        max_low_memory_side=768,
        max_low_memory_steps=20,
        live_proof=False,
    ),
}


def _flux_execution_profile(
    profile_id: str,
    model_type: str,
    modes: tuple[str, ...],
    pipeline_class: str,
    repo: str,
    *,
    live_proof: bool = False,
) -> DiffusersExecutionProfile:
    """Build the shared generic-image execution contract for FLUX variants."""

    return DiffusersExecutionProfile(
        id=profile_id,
        model_type=model_type,
        modes=modes,
        backend_path="modules.DiffusersImage.LoadPipeline",
        pipeline_class=pipeline_class,
        default_repo=repo,
        fallback_repo=None,
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
        live_proof=live_proof,
    )


DIFFUSERS_EXECUTION_PROFILES.update(
    {
        "flux2-klein:direct": _flux_execution_profile(
            "flux2-klein:direct",
            "Flux2KleinPipeline",
            ("text_to_image", "edit_image", "multi_image_reference_edit"),
            "Flux2KleinPipeline",
            FLUX2_KLEIN_REPO,
            live_proof=True,
        ),
        "flux-krea:direct": _flux_execution_profile(
            "flux-krea:direct", "FluxKreaPipeline", ("text_to_image",), "FluxPipeline", FLUX_KREA_REPO
        ),
        "flux-kontext:direct": _flux_execution_profile(
            "flux-kontext:direct",
            "FluxKontextPipeline",
            ("edit_image", "multi_image_reference_edit"),
            "FluxKontextPipeline",
            FLUX_KONTEXT_REPO,
        ),
        "flux-fill:direct": _flux_execution_profile(
            "flux-fill:direct", "FluxFillPipeline", ("inpaint", "outpaint"), "FluxFillPipeline", FLUX_FILL_REPO
        ),
        "flux-depth:direct": _flux_execution_profile(
            "flux-depth:direct", "FluxDepthPipeline", ("control_image",), "FluxControlPipeline", FLUX_DEPTH_REPO
        ),
        "flux-canny:direct": _flux_execution_profile(
            "flux-canny:direct", "FluxCannyPipeline", ("control_image",), "FluxControlPipeline", FLUX_CANNY_REPO
        ),
        "flux-redux:direct": _flux_execution_profile(
            "flux-redux:direct",
            "FluxReduxPipeline",
            ("edit_image",),
            "FluxReduxPipeline",
            FLUX_REDUX_REPO,
        ),
    }
)

EXPERIMENTAL_DIFFUSERS_PIPELINES = [
    {
        "modelType": "StableDiffusionXLModularPipeline",
        "label": "Stable Diffusion XL (Modular)",
        "mediaKind": "image",
        "pipelineClasses": ["StableDiffusionXLModularPipeline"],
        "runnableModes": ["text_to_image", "image_to_image", "inpaint", "control_image"],
    },
    {
        "modelType": "FluxModularPipeline",
        "label": "FLUX (Modular)",
        "mediaKind": "image",
        "pipelineClasses": ["FluxModularPipeline"],
        "runnableModes": ["text_to_image", "image_to_image", "control_image"],
    },
    {
        "modelType": "Flux2KleinModularPipeline",
        "label": "FLUX.2 Klein (Modular)",
        "mediaKind": "image",
        "defaultRepo": "black-forest-labs/FLUX.2-klein-4B",
        "pipelineClasses": ["Flux2KleinPipeline", "Flux2KleinModularPipeline"],
        "backendPath": "modules.DiffusersImage.LoadPipeline",
        "runnableModes": ["text_to_image", "edit_image", "multi_image_reference_edit"],
    },
    {
        "modelType": "WanModularPipeline",
        "label": "Wan Text to Video (Modular)",
        "mediaKind": "video",
        "pipelineClasses": ["WanModularPipeline"],
        "runnableModes": ["text_to_video"],
    },
    {
        "modelType": "WanImage2VideoModularPipeline",
        "label": "Wan Image to Video (Modular)",
        "mediaKind": "video",
        "pipelineClasses": ["WanImage2VideoModularPipeline"],
        "runnableModes": ["image_to_video"],
    },
]


def public_experimental_pipelines() -> list[dict]:
    parameter_aliases = {
        "modelRepository": ["model_id", "model", "repo"],
        "guidanceScale": ["guidance_scale", "true_cfg_scale", "guidance"],
        "steps": ["num_inference_steps", "steps"],
        "sourceImage": ["image", "reference_images"],
        "maskImage": ["mask_image", "mask"],
        "controlImage": ["control_image", "conditioning_image"],
    }
    return [
        {
            **pipeline,
            "schemaVersion": 2,
            "supportTier": "experimental",
            "executionProfiles": [],
            "inputContracts": pipeline.get("inputContracts", {}),
            "parameterAliases": parameter_aliases,
            "defaults": pipeline.get("defaults", {}),
            "artifactCandidates": [pipeline["defaultRepo"]] if pipeline.get("defaultRepo") else [],
            "revisionCandidates": pipeline.get("revisionCandidates", []),
            "quantizationSupport": pipeline.get(
                "quantizationSupport",
                {"defaultMode": "none", "components": [], "offloadModes": []},
            ),
            "qualificationStatus": pipeline.get("qualificationStatus", "unqualified"),
        }
        for pipeline in EXPERIMENTAL_DIFFUSERS_PIPELINES
    ]


def public_execution_profiles() -> list[dict]:
    return [profile.to_public_dict() for profile in DIFFUSERS_EXECUTION_PROFILES.values()]
