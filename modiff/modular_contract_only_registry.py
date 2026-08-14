"""Data-only registry for pinned Modular classes that are not executable yet."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContractOnlyModularPipeline:
    class_name: str
    label: str
    batch: str
    aliases: tuple[tuple[str, str], ...] = ()


CURRENT_PIN_CONTRACT_ONLY_MODULAR_IMAGE_PIPELINES = (
    ContractOnlyModularPipeline("AnimaModularPipeline", "Anima (Contract only)", "image"),
    ContractOnlyModularPipeline("Flux2ModularPipeline", "FLUX.2 (Contract only)", "image"),
    ContractOnlyModularPipeline(
        "Flux2KleinBaseModularPipeline",
        "FLUX.2 Klein Base (Contract only)",
        "image",
    ),
    ContractOnlyModularPipeline("Ideogram4ModularPipeline", "Ideogram 4 (Contract only)", "image"),
    ContractOnlyModularPipeline("Krea2ModularPipeline", "Krea 2 (Contract only)", "image"),
    ContractOnlyModularPipeline(
        "Krea2TurboModularPipeline",
        "Krea 2 Turbo (Contract only)",
        "image",
    ),
    ContractOnlyModularPipeline(
        "StableDiffusion3ModularPipeline",
        "Stable Diffusion 3 (Contract only)",
        "image",
    ),
)

CURRENT_PIN_CONTRACT_ONLY_MODULAR_VIDEO_PIPELINES = (
    ContractOnlyModularPipeline("HeliosModularPipeline", "Helios (Contract only)", "video"),
    ContractOnlyModularPipeline(
        "HeliosPyramidModularPipeline",
        "Helios Pyramid (Contract only)",
        "video",
    ),
    ContractOnlyModularPipeline(
        "HeliosPyramidDistilledModularPipeline",
        "Helios Pyramid Distilled (Contract only)",
        "video",
    ),
    ContractOnlyModularPipeline(
        "HunyuanVideo15ModularPipeline",
        "HunyuanVideo 1.5 (Contract only)",
        "video",
    ),
    ContractOnlyModularPipeline(
        "WanAnimate2ModularPipeline",
        "Wan Animate 2 (Contract only)",
        "video",
        (("default", "character_animate"),),
    ),
    ContractOnlyModularPipeline(
        "WanAnimate2DistilledModularPipeline",
        "Wan Animate 2 Distilled (Contract only)",
        "video",
        (("default", "character_animate"),),
    ),
)

CURRENT_PIN_CONTRACT_ONLY_MODULAR_AUDIO_PIPELINES = (
    ContractOnlyModularPipeline(
        "MiniMaxMusic3ModularPipeline",
        "MiniMax Music 3 (Contract only)",
        "audio",
        (("default", "text_to_audio"),),
    ),
)

# Cosmos can cross image/video and, for Omni, sound/action domains.
CURRENT_PIN_CONTRACT_ONLY_MODULAR_MULTIMODAL_PIPELINES = (
    ContractOnlyModularPipeline(
        "Cosmos3DistilledModularPipeline",
        "Cosmos 3 Distilled (Contract only)",
        "multimodal",
    ),
    ContractOnlyModularPipeline(
        "Cosmos3OmniModularPipeline",
        "Cosmos 3 Omni (Contract only)",
        "multimodal",
        (
            ("text2video_with_sound", "text_to_video_with_audio"),
            ("image2video_with_sound", "image_to_video_with_audio"),
            ("video2video_with_sound", "video_to_video_with_audio"),
        ),
    ),
    ContractOnlyModularPipeline(
        "MiniMaxH3ModularPipeline",
        "MiniMax H3 (Contract only)",
        "multimodal",
        (
            ("t2va", "text_to_video_with_audio"),
            ("fl2va", "first_last_frame_to_video_with_audio"),
            ("ref2va", "reference_to_video_with_audio"),
        ),
    ),
    ContractOnlyModularPipeline(
        "LTX2ModularPipeline",
        "LTX-2 (Contract only)",
        "multimodal",
        (
            ("text2video", "text_to_video_with_audio"),
            ("image2video", "image_to_video_with_audio"),
            ("condition", "condition_to_video_with_audio"),
            ("in_context", "in_context_to_video_with_audio"),
        ),
    ),
    ContractOnlyModularPipeline(
        "LTX25ModularPipeline",
        "LTX-2.5 (Contract only)",
        "multimodal",
        (
            ("text2video", "text_to_video_with_audio"),
            ("image2video", "image_to_video_with_audio"),
            ("condition", "condition_to_video_with_audio"),
            ("in_context", "in_context_to_video_with_audio"),
        ),
    ),
)

CURRENT_PIN_CONTRACT_ONLY_MODULAR_PIPELINES = (
    *CURRENT_PIN_CONTRACT_ONLY_MODULAR_IMAGE_PIPELINES,
    *CURRENT_PIN_CONTRACT_ONLY_MODULAR_VIDEO_PIPELINES,
    *CURRENT_PIN_CONTRACT_ONLY_MODULAR_AUDIO_PIPELINES,
    *CURRENT_PIN_CONTRACT_ONLY_MODULAR_MULTIMODAL_PIPELINES,
)

CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME = {
    item.class_name: item for item in CURRENT_PIN_CONTRACT_ONLY_MODULAR_PIPELINES
}

# These exact pinned Modular exports add no public task surface beyond the
# already executable standard classes listed here. Keep them separate from the
# contract-only registry: equivalence is a reviewed routing decision, not a
# runnable Modular profile or loader claim.
CURRENT_PIN_EQUIVALENT_MODULAR_TARGETS = {
    "ErnieImageModularPipeline": ("ErnieImagePipeline",),
    "LTXModularPipeline": ("LTXConditionPipeline",),
    "Wan22ModularPipeline": ("WanPipeline",),
    "Wan22Image2VideoModularPipeline": ("WanImageToVideoPipeline",),
}
