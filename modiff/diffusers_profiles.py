from __future__ import annotations

from dataclasses import asdict, dataclass

from modiff.diffusers_offload import (
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
FLUX_REDUX_REPO = "black-forest-labs/FLUX.1-Redux-dev"


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
        backend_path="modules.QwenImage.LoadPipeline",
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
        backend_path="modules.QwenImage.LoadInpaintPipeline",
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
            "image_to_video",
            "video_to_video",
            "video_inpaint",
            "video_outpaint",
            "reference_to_video",
            "control_to_video",
            "video_color_edit",
        ),
        backend_path="modules.WanVACE.LoadPipeline",
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


def public_execution_profiles() -> list[dict]:
    return [profile.to_public_dict() for profile in DIFFUSERS_EXECUTION_PROFILES.values()]
