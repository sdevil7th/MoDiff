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
    ContractOnlyModularPipeline("ErnieImageModularPipeline", "ERNIE Image (Contract only)", "image"),
    ContractOnlyModularPipeline("Flux2ModularPipeline", "FLUX.2 (Contract only)", "image"),
    ContractOnlyModularPipeline(
        "Flux2KleinBaseModularPipeline",
        "FLUX.2 Klein Base (Contract only)",
        "image",
    ),
    ContractOnlyModularPipeline("Ideogram4ModularPipeline", "Ideogram 4 (Contract only)", "image"),
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
    ContractOnlyModularPipeline("LTXModularPipeline", "LTX Video (Contract only)", "video"),
    ContractOnlyModularPipeline(
        "Wan22ModularPipeline",
        "Wan 2.2 Text to Video (Contract only)",
        "video",
        (("default", "text_to_video"),),
    ),
    ContractOnlyModularPipeline(
        "Wan22Image2VideoModularPipeline",
        "Wan 2.2 Image to Video (Contract only)",
        "video",
        (("default", "image_to_video"),),
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
)

CURRENT_PIN_CONTRACT_ONLY_MODULAR_PIPELINES = (
    *CURRENT_PIN_CONTRACT_ONLY_MODULAR_IMAGE_PIPELINES,
    *CURRENT_PIN_CONTRACT_ONLY_MODULAR_VIDEO_PIPELINES,
    *CURRENT_PIN_CONTRACT_ONLY_MODULAR_MULTIMODAL_PIPELINES,
)

CURRENT_PIN_CONTRACT_ONLY_MODULAR_BY_NAME = {
    item.class_name: item for item in CURRENT_PIN_CONTRACT_ONLY_MODULAR_PIPELINES
}
