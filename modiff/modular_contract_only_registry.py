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

CURRENT_PIN_CONTRACT_ONLY_MODULAR_VIDEO_PIPELINES = ()

CURRENT_PIN_CONTRACT_ONLY_MODULAR_AUDIO_PIPELINES = ()

# Cosmos can cross image/video and, for Omni, sound/action domains.
CURRENT_PIN_CONTRACT_ONLY_MODULAR_MULTIMODAL_PIPELINES = (
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

# Already-promoted whole-workflow adapters without an entry in the split-node
# truth matrix. Discovery must retain them when they leave the contract-only
# list. This is generator coverage, not an additional execution permission.
CURRENT_PIN_PROMOTED_MODULAR_DISCOVERY = {
    "Cosmos3DistilledModularPipeline": {
        "text2image": "text_to_image", "text2video": "text_to_video",
        "image2video": "image_to_video", "video2video": "video_to_video",
    },
    "MiniMaxH3ModularPipeline": {
        "t2va": "text_to_video_with_audio",
        "fl2va": "first_last_frame_to_video_with_audio",
        "ref2va": "reference_to_video_with_audio",
    },
}

# These exact pinned Modular exports add no public task surface beyond the
# already executable standard classes listed here. Keep them separate from the
# contract-only registry: equivalence is a reviewed routing decision, not a
# runnable Modular profile or loader claim.
CURRENT_PIN_EQUIVALENT_MODULAR_TARGETS = {
    "ErnieImageModularPipeline": ("ErnieImagePipeline",),
    "LTXModularPipeline": ("LTXConditionPipeline",),
    "LTX2ModularPipeline": ("LTX2ConditionPipeline", "LTX2InContextPipeline"),
    "Wan22ModularPipeline": ("WanPipeline",),
    "Wan22Image2VideoModularPipeline": ("WanImageToVideoPipeline",),
}

# Some upstream AutoBlocks expose more workflows than the reviewed standard
# executor can represent exactly.  Keep those promotions workflow-scoped so a
# proven text/image route cannot relabel sibling condition or in-context
# workflows as equivalent.
CURRENT_PIN_EQUIVALENT_MODULAR_WORKFLOW_TARGETS = {
    ("LTX2ModularPipeline", "text2video"): ("LTX2ConditionPipeline",),
    ("LTX2ModularPipeline", "image2video"): ("LTX2ConditionPipeline",),
    # The reviewed adapter materializes an ordered image list as official
    # LTX2VideoCondition instances before calling the exact condition
    # pipeline.  In-context remains separate because it additionally requires
    # LTX2ReferenceCondition values and a pinned IC-LoRA.
    ("LTX2ModularPipeline", "condition"): ("LTX2ConditionPipeline",),
    ("LTX2ModularPipeline", "in_context"): ("LTX2InContextPipeline",),
}


def equivalent_modular_targets(pipeline_class: str, workflow_id: str) -> tuple[str, ...]:
    return CURRENT_PIN_EQUIVALENT_MODULAR_WORKFLOW_TARGETS.get(
        (pipeline_class, workflow_id),
        CURRENT_PIN_EQUIVALENT_MODULAR_TARGETS.get(pipeline_class, ()),
    )
