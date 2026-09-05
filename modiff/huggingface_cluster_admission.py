"""Static admission gate for executable Diffusers Cluster Node candidates.

The first-party Hugging Face library is intentionally broader than MoDiff's
executable graph surface.  This module joins one exact upstream workflow and
graph-adapter contract to one existing Studio execution specification.  It is
data-only and performs no Diffusers, Transformers, Torch, model, or node
registry import.

Passing this gate means that the reviewed Studio graph can materialize the
declared generic Modular Diffusers actions and state edges for the selected
workflow.  It is not live model evidence, Auto eligibility, or permission to
publish the Cluster Node by itself.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from modiff.huggingface_cluster_promotions import promotion_receipt_for_admission
from modiff.model_artifact_catalog import catalog_repository_pin, require_catalog_revision
from modiff.modular_contract_only_registry import equivalent_modular_targets
from modiff.modular_workflow_contracts import (
    WAN_T2V_14B_REPOSITORY,
    WAN_T2V_REPOSITORY,
    WAN_WORKFLOW_REPOSITORIES,
)
from modiff.studio_execution_specs import (
    STUDIO_EXECUTION_SPEC_DEFINITIONS,
    studio_execution_spec_for_pair,
    studio_model_dependencies_for_pair,
)


CLUSTER_EXECUTION_ADMISSION_SCHEMA_VERSION = 4

_MODELS_LOADER = "modules.ModularDiffusers.ModelsLoader"
_DIRECT_LOADER_CONTRACTS = {
    "direct-diffusers-image": ("modules.DiffusersImage", "LoadPipeline"),
    "direct-diffusers-video": ("modules.DiffusersVideo", "LoadPipeline"),
}
# Wan 2.2 is the only reviewed local compatibility alias in this slice.  The
# adapter resolves it to upstream WanPipeline; keeping the alias here makes the
# exception explicit instead of treating arbitrary pipeline names as
# equivalent.
_EQUIVALENT_STANDARD_EXECUTION_CLASSES = {
    "Flux2ModularPipeline": "Flux2Pipeline",
    "ErnieImageModularPipeline": "ErnieImagePipeline",
    "LTXModularPipeline": "LTXConditionPipeline",
    "Wan22ModularPipeline": "Wan22Pipeline",
    "Wan22Image2VideoModularPipeline": "WanImageToVideoPipeline",
    "LTX2ModularPipeline": "LTX2ConditionPipeline",
}
_EQUIVALENT_STANDARD_WORKFLOW_EXECUTION_CLASSES = {
    ("LTX2ModularPipeline", "in_context"): "LTX2InContextPipeline",
}


def _equivalent_standard_execution_class(pipeline_class: str, workflow_id: str) -> str | None:
    return _EQUIVALENT_STANDARD_WORKFLOW_EXECUTION_CLASSES.get(
        (pipeline_class, workflow_id),
        _EQUIVALENT_STANDARD_EXECUTION_CLASSES.get(pipeline_class),
    )


_ACTION_ROLE_CONTRACTS = {
    "text_encoder": ("prompt", "modules.ModularDiffusers.EncodePrompt"),
    "image_encoder": ("imageEmbeddings", "modules.ModularDiffusers.ImageEmbeddings"),
    "vae_encoder": ("imageEncode", "modules.ModularDiffusers.ImageEncode"),
    "controlnet": ("controlnet", "modules.ModularDiffusers.Controlnet"),
    "ip_adapter": ("ipAdapter", "modules.ModularDiffusers.IPAdapter"),
    "denoise": ("denoise", "modules.ModularDiffusers.Denoise"),
    "decoder": ("decode", "modules.ModularDiffusers.DecodeLatents"),
    "semantic_generator": ("prompt", "modules.ModularDiffusers.WorkflowSemanticGeneration"),
    "workflow_denoise": ("denoise", "modules.ModularDiffusers.WorkflowDenoise"),
    "workflow_audio_decoder": ("decode", "modules.ModularDiffusers.WorkflowDecodeAudio"),
    "workflow_text_encoder": ("prompt", "modules.ModularDiffusers.WorkflowTextEncode"),
    "workflow_image_encoder": ("imageEncode", "modules.ModularDiffusers.WorkflowImageEncode"),
    "workflow_video_image_encoder": (
        "imageEncode",
        "modules.ModularDiffusers.WorkflowVideoImageEncode",
    ),
    "workflow_video_encoder": ("videoEncode", "modules.ModularDiffusers.WorkflowVideoEncode"),
    "workflow_image_denoise": ("denoise", "modules.ModularDiffusers.WorkflowImageDenoise"),
    "workflow_image_decoder": ("decode", "modules.ModularDiffusers.WorkflowDecodeImage"),
    "workflow_video_denoise": ("denoise", "modules.ModularDiffusers.WorkflowVideoDenoise"),
    "workflow_video_decoder": ("decode", "modules.ModularDiffusers.WorkflowDecodeVideo"),
    "workflow_hunyuan_video15_text_encoder": (
        "prompt",
        "modules.ModularDiffusers.WorkflowHunyuanVideo15TextEncode",
    ),
    "workflow_hunyuan_video15_vae_encoder": (
        "imageEncode",
        "modules.ModularDiffusers.WorkflowHunyuanVideo15VaeEncode",
    ),
    "workflow_hunyuan_video15_image_encoder": (
        "imageEmbeddings",
        "modules.ModularDiffusers.WorkflowHunyuanVideo15ImageEncode",
    ),
    "workflow_hunyuan_video15_denoise": (
        "denoise",
        "modules.ModularDiffusers.WorkflowHunyuanVideo15Denoise",
    ),
    "workflow_hunyuan_video15_decoder": (
        "decode",
        "modules.ModularDiffusers.WorkflowHunyuanVideo15Decode",
    ),
    "workflow_stable_diffusion3_text_encoder": (
        "prompt",
        "modules.ModularDiffusers.WorkflowStableDiffusion3TextEncode",
    ),
    "workflow_stable_diffusion3_vae_encoder": (
        "imageEncode",
        "modules.ModularDiffusers.WorkflowStableDiffusion3VaeEncode",
    ),
    "workflow_stable_diffusion3_denoise": (
        "denoise",
        "modules.ModularDiffusers.WorkflowStableDiffusion3Denoise",
    ),
    "workflow_stable_diffusion3_decoder": (
        "decode",
        "modules.ModularDiffusers.WorkflowStableDiffusion3Decode",
    ),
    "workflow_krea2_text_encoder": (
        "prompt",
        "modules.ModularDiffusers.WorkflowKrea2TextEncode",
    ),
    "workflow_krea2_turbo_text_encoder": (
        "prompt",
        "modules.ModularDiffusers.WorkflowKrea2TurboTextEncode",
    ),
    "workflow_krea2_denoise": (
        "denoise",
        "modules.ModularDiffusers.WorkflowKrea2Denoise",
    ),
    "workflow_krea2_turbo_denoise": (
        "denoise",
        "modules.ModularDiffusers.WorkflowKrea2TurboDenoise",
    ),
    "workflow_krea2_decoder": (
        "decode",
        "modules.ModularDiffusers.WorkflowKrea2Decode",
    ),
    "workflow_ideogram4_prompt_upsample": (
        "prompt",
        "modules.ModularDiffusers.WorkflowIdeogram4PromptUpsample",
    ),
    "workflow_ideogram4_text_encoder": (
        "textEncode",
        "modules.ModularDiffusers.WorkflowIdeogram4TextEncode",
    ),
    "workflow_ideogram4_denoise": (
        "denoise",
        "modules.ModularDiffusers.WorkflowIdeogram4Denoise",
    ),
    "workflow_ideogram4_decoder": (
        "decode",
        "modules.ModularDiffusers.WorkflowIdeogram4Decode",
    ),
    "workflow_cosmos3_distilled_text_encoder": (
        "prompt",
        "modules.ModularDiffusers.WorkflowCosmos3DistilledTextEncode",
    ),
    "workflow_cosmos3_distilled_vae_encoder": (
        "imageEncode",
        "modules.ModularDiffusers.WorkflowCosmos3DistilledVaeEncode",
    ),
    "workflow_cosmos3_distilled_denoise": (
        "denoise",
        "modules.ModularDiffusers.WorkflowCosmos3DistilledDenoise",
    ),
    "workflow_cosmos3_distilled_decoder": (
        "decode",
        "modules.ModularDiffusers.WorkflowCosmos3DistilledDecode",
    ),
    "workflow_cosmos3_omni_text_encoder": (
        "prompt",
        "modules.ModularDiffusers.WorkflowCosmos3OmniTextEncode",
    ),
    "workflow_cosmos3_omni_vae_encoder": (
        "imageEncode",
        "modules.ModularDiffusers.WorkflowCosmos3OmniVaeEncode",
    ),
    "workflow_cosmos3_omni_denoise": (
        "denoise",
        "modules.ModularDiffusers.WorkflowCosmos3OmniDenoise",
    ),
    "workflow_cosmos3_omni_decoder": (
        "decode",
        "modules.ModularDiffusers.WorkflowCosmos3OmniDecode",
    ),
    "workflow_cosmos3_omni_after_decode": (
        "afterDecode",
        "modules.ModularDiffusers.WorkflowCosmos3OmniAfterDecode",
    ),
    "workflow_minimax_h3_before_encode": (
        "beforeEncode",
        "modules.ModularDiffusers.WorkflowMiniMaxH3BeforeEncode",
    ),
    "workflow_minimax_h3_text_encoder": (
        "prompt",
        "modules.ModularDiffusers.WorkflowMiniMaxH3TextEncode",
    ),
    "workflow_minimax_h3_vae_encoder": (
        "imageEncode",
        "modules.ModularDiffusers.WorkflowMiniMaxH3VaeEncode",
    ),
    "workflow_minimax_h3_denoise": (
        "denoise",
        "modules.ModularDiffusers.WorkflowMiniMaxH3Denoise",
    ),
    "workflow_minimax_h3_decoder": (
        "decode",
        "modules.ModularDiffusers.WorkflowMiniMaxH3Decode",
    ),
    "workflow_ltx25_text_encoder": (
        "prompt",
        "modules.ModularDiffusers.WorkflowLTX25TextEncode",
    ),
    "workflow_ltx25_duration": (
        "duration",
        "modules.ModularDiffusers.WorkflowLTX25Duration",
    ),
    "workflow_ltx25_vae_encoder": (
        "imageEncode",
        "modules.ModularDiffusers.WorkflowLTX25VaeEncode",
    ),
    "workflow_ltx25_condition_encoder": (
        "conditionEncode",
        "modules.ModularDiffusers.WorkflowLTX25ConditionEncode",
    ),
    "workflow_ltx25_reference_encoder": (
        "referenceEncode",
        "modules.ModularDiffusers.WorkflowLTX25ReferenceEncode",
    ),
    "workflow_ltx25_denoise": (
        "denoise",
        "modules.ModularDiffusers.WorkflowLTX25Denoise",
    ),
    "workflow_ltx25_decoder": (
        "decode",
        "modules.ModularDiffusers.WorkflowLTX25Decode",
    ),
    "workflow_wan_animate_text_encoder": (
        "prompt",
        "modules.ModularDiffusers.WorkflowWanAnimateTextEncode",
    ),
    "workflow_wan_animate_image_encoder": (
        "imageEmbeddings",
        "modules.ModularDiffusers.WorkflowWanAnimateImageEncode",
    ),
    "workflow_wan_animate_video_encoder": (
        "videoEncode",
        "modules.ModularDiffusers.WorkflowWanAnimateVideoEncode",
    ),
    "workflow_wan_animate_vae_encoder": (
        "imageEncode",
        "modules.ModularDiffusers.WorkflowWanAnimateVaeEncode",
    ),
    "workflow_wan_animate_denoise": (
        "denoise",
        "modules.ModularDiffusers.WorkflowWanAnimateDenoise",
    ),
    "workflow_wan_animate_decoder": (
        "decode",
        "modules.ModularDiffusers.WorkflowWanAnimateDecode",
    ),
}
_AUXILIARY_NODE_KEYS = {
    _MODELS_LOADER,
    "modules.ModularDiffusers.AutoModelLoader",
    "modules.ModularDiffusers.Guider",
    "modules.Image.Load",
    "modules.Image.Preview",
    "modules.Video.Load",
    "modules.Video.Export",
    "modules.Video.ExportWithAudio",
    "modules.Audio.Export",
}
_UPSTREAM_INPUT_BY_BINDING_SOURCE = {
    "prompt": "prompt",
    "lyrics": "lyrics",
    "referenceImages": "image",
    "conditionImages": "conditions",
    "referenceVideos": "reference_conditions",
    "sourceVideo": "video",
    "poseVideo": "driving_video",
    "requiredNumFrames": "num_frames",
    "maskImage": "mask_image",
    "lastImage": "last_image",
    "controlImage": "control_image",
    "ipAdapterImage": "ip_adapter_image",
    "controlMode": "control_mode",
    "layers": "layers",
    "references": "references",
}
# Some official workflows expose a required execution parameter that older
# MoDiff adapters intentionally treated as an optional instance control. Keep
# that projection route-specific: globally mapping ``steps`` would incorrectly
# strengthen unrelated Anima and Helios contracts.
_PIPELINE_UPSTREAM_INPUT_BY_BINDING_SOURCE = {
    ("Cosmos3OmniModularPipeline", "steps"): "num_inference_steps",
    ("MiniMaxH3ModularPipeline", "steps"): "num_inference_steps",
}
_OPTIONAL_UPSTREAM_INPUTS = {
    # The official FL2VA workflow accepts first, last, or both keyframes. Its
    # adapter therefore has neither individual field in requiredInputs, while
    # the exact Cluster must still expose both optional public boundaries.
    ("MiniMaxH3ModularPipeline", "fl2va"): frozenset({"image", "last_image"}),
}
_INSTANCE_INPUT_BY_BINDING_SOURCE = {
    "dtype": "dtype",
    "prompt": "prompt",
    "negativePrompt": "negative_prompt",
    "promptRef": "prompt_ref",
    "referenceImages": "image",
    "conditionImages": "conditions",
    "referenceVideos": "reference_conditions",
    "sourceVideo": "video",
    "poseVideo": "driving_video",
    "requiredNumFrames": "num_frames",
    "maskImage": "mask_image",
    "lastImage": "last_image",
    "controlImage": "control_image",
    "ipAdapterImage": "ip_adapter_image",
    "controlMode": "control_mode",
    "controlGuidanceStart": "control_guidance_start",
    "controlGuidanceEnd": "control_guidance_end",
    "width": "width",
    "height": "height",
    "optionalWidth": "width",
    "optionalHeight": "height",
    "steps": "num_inference_steps",
    "numFrames": "num_frames",
    "maxSequenceLength": "max_sequence_length",
    "segmentFrameLength": "segment_frame_length",
    "previousConditioningFrames": "prev_segment_conditioning_frames",
    "layers": "layers",
    "conditioningScale": "controlnet_conditioning_scale",
    "outputType": "output_type",
    "resolution": "resolution",
    "strength": "strength",
    "paddingMaskCrop": "padding_mask_crop",
    "lyrics": "lyrics",
    "audioDuration": "audio_duration",
    "references": "references",
}
_SEALED_BINDING_SOURCES = {
    "artifact",
    "pipelineClass",
    "executionProfileId",
    "defaultRevision",
    "kind",
    "repo",
    "revision",
    "empty",
    "true",
    "false",
    "addAlpha",
    "removeAlpha",
    "ordinary",
    "fp16",
    "controlnetKind",
    "controlnetRepo",
    "controlnetRevision",
    "controlnetWeightVariant",
    "controlnetRouteVariant",
    "controlnetLoadClass",
    "ipAdapterRepo",
    "ipAdapterRevision",
    "ipAdapterWeightName",
    "adapterWeightName",
    "classifierFreeGuidance",
    "mode",
    "defaultWorkflow",
    "workflowId",
    "workflowTextEncoderBlock",
    "workflowBeforeEncodeBlock",
    "workflowImageEncoderBlock",
    "workflowVaeEncoderBlock",
    "workflowImageEmbeddingsBlock",
    "semanticGeneratorBlock",
    "workflowDenoiseBlock",
    "workflowDecodeBlock",
    "workflowAfterDecodeBlock",
    "oneFrame",
    "oneVideo",
    "sampleRate44100",
    "distilledSteps4",
    "distilledGuidance1",
}
_DYNAMIC_FIELD_ACTIONS = (
    ("models", "model_type", "onChange", "pipelineClass"),
    ("prompt", "text_encoders", "onSignal", "pipelineClass"),
    ("imageEmbeddings", "image_encoder", "onSignal", "pipelineClass"),
    ("imageEncode", "vae", "onSignal", "pipelineClass"),
    ("denoise", "unet", "onSignal", "pipelineClass"),
    ("decode", "vae", "onSignal", "pipelineClass"),
    ("controlnet", "controlnet_bundle", "onSignal", "pipelineClass"),
    ("controlnetModel", "model_type", "onChange", "kind"),
    ("controlnetModel", "model_type", "onChange", "controlnetKind"),
    ("ipAdapter", "unet", "onSignal", "pipelineClass"),
    ("guider", "guider_out", "onSignal", "pipelineClass"),
)
_REQUIRED_LOADER_BINDINGS = {
    ("models", "model_type", "pipelineClass"),
    ("models", "repo_id", "artifact"),
    ("models", "trust_remote_code", "false"),
}
_REQUIRED_COMPONENT_EDGES = {
    "text_encoder": {("models", "text_encoders", "prompt", "text_encoders")},
    "image_encoder": {("models", "image_encoder", "imageEmbeddings", "image_encoder")},
    "vae_encoder": {("models", "vae_out", "imageEncode", "vae")},
    "controlnet": {("controlnetModel", "model", "controlnet", "controlnet")},
    "ip_adapter": {
        ("models", "unet_out", "ipAdapter", "unet"),
        ("guider", "guider_out", "ipAdapter", "guider"),
        ("guider", "guider_out", "denoise", "guider"),
    },
    "denoise": {
        ("models", "unet_out", "denoise", "unet"),
        ("models", "scheduler", "denoise", "scheduler"),
    },
    "decoder": {("models", "vae_out", "decode", "vae")},
    "semantic_generator": {("models", "pipeline_components", "prompt", "pipeline_components")},
    "workflow_denoise": {("models", "pipeline_components", "denoise", "pipeline_components")},
    "workflow_audio_decoder": {("models", "pipeline_components", "decode", "pipeline_components")},
    "workflow_text_encoder": {("models", "pipeline_components", "prompt", "pipeline_components")},
    "workflow_image_encoder": {("models", "pipeline_components", "imageEncode", "pipeline_components")},
    "workflow_video_image_encoder": {("models", "pipeline_components", "imageEncode", "pipeline_components")},
    "workflow_video_encoder": {("models", "pipeline_components", "videoEncode", "pipeline_components")},
    "workflow_image_denoise": {("models", "pipeline_components", "denoise", "pipeline_components")},
    "workflow_image_decoder": {("models", "pipeline_components", "decode", "pipeline_components")},
    "workflow_video_denoise": {("models", "pipeline_components", "denoise", "pipeline_components")},
    "workflow_video_decoder": {("models", "pipeline_components", "decode", "pipeline_components")},
    "workflow_hunyuan_video15_text_encoder": {("models", "pipeline_components", "prompt", "pipeline_components")},
    "workflow_hunyuan_video15_vae_encoder": {("models", "pipeline_components", "imageEncode", "pipeline_components")},
    "workflow_hunyuan_video15_image_encoder": {
        ("models", "pipeline_components", "imageEmbeddings", "pipeline_components")
    },
    "workflow_hunyuan_video15_denoise": {("models", "pipeline_components", "denoise", "pipeline_components")},
    "workflow_hunyuan_video15_decoder": {("models", "pipeline_components", "decode", "pipeline_components")},
    "workflow_stable_diffusion3_text_encoder": {("models", "pipeline_components", "prompt", "pipeline_components")},
    "workflow_stable_diffusion3_vae_encoder": {
        ("models", "pipeline_components", "imageEncode", "pipeline_components")
    },
    "workflow_stable_diffusion3_denoise": {("models", "pipeline_components", "denoise", "pipeline_components")},
    "workflow_stable_diffusion3_decoder": {("models", "pipeline_components", "decode", "pipeline_components")},
    "workflow_krea2_text_encoder": {("models", "pipeline_components", "prompt", "pipeline_components")},
    "workflow_krea2_turbo_text_encoder": {("models", "pipeline_components", "prompt", "pipeline_components")},
    "workflow_krea2_denoise": {("models", "pipeline_components", "denoise", "pipeline_components")},
    "workflow_krea2_turbo_denoise": {("models", "pipeline_components", "denoise", "pipeline_components")},
    "workflow_krea2_decoder": {("models", "pipeline_components", "decode", "pipeline_components")},
    "workflow_ideogram4_prompt_upsample": {("models", "pipeline_components", "prompt", "pipeline_components")},
    "workflow_ideogram4_text_encoder": {("models", "pipeline_components", "textEncode", "pipeline_components")},
    "workflow_ideogram4_denoise": {("models", "pipeline_components", "denoise", "pipeline_components")},
    "workflow_ideogram4_decoder": {("models", "pipeline_components", "decode", "pipeline_components")},
    "workflow_cosmos3_distilled_text_encoder": {("models", "pipeline_components", "prompt", "pipeline_components")},
    "workflow_cosmos3_distilled_vae_encoder": {
        ("models", "pipeline_components", "imageEncode", "pipeline_components")
    },
    "workflow_cosmos3_distilled_denoise": {("models", "pipeline_components", "denoise", "pipeline_components")},
    "workflow_cosmos3_distilled_decoder": {("models", "pipeline_components", "decode", "pipeline_components")},
    "workflow_cosmos3_omni_text_encoder": {("models", "pipeline_components", "prompt", "pipeline_components")},
    "workflow_cosmos3_omni_vae_encoder": {("models", "pipeline_components", "imageEncode", "pipeline_components")},
    "workflow_cosmos3_omni_denoise": {("models", "pipeline_components", "denoise", "pipeline_components")},
    "workflow_cosmos3_omni_decoder": {("models", "pipeline_components", "decode", "pipeline_components")},
    "workflow_cosmos3_omni_after_decode": {("models", "pipeline_components", "afterDecode", "pipeline_components")},
    "workflow_minimax_h3_before_encode": {("models", "pipeline_components", "beforeEncode", "pipeline_components")},
    "workflow_minimax_h3_text_encoder": {("models", "pipeline_components", "prompt", "pipeline_components")},
    "workflow_minimax_h3_vae_encoder": {("models", "pipeline_components", "imageEncode", "pipeline_components")},
    "workflow_minimax_h3_denoise": {("models", "pipeline_components", "denoise", "pipeline_components")},
    "workflow_minimax_h3_decoder": {("models", "pipeline_components", "decode", "pipeline_components")},
    "workflow_ltx25_text_encoder": {("models", "pipeline_components", "prompt", "pipeline_components")},
    "workflow_ltx25_duration": {("models", "pipeline_components", "duration", "pipeline_components")},
    "workflow_ltx25_vae_encoder": {("models", "pipeline_components", "imageEncode", "pipeline_components")},
    "workflow_ltx25_condition_encoder": {("models", "pipeline_components", "conditionEncode", "pipeline_components")},
    "workflow_ltx25_reference_encoder": {("models", "pipeline_components", "referenceEncode", "pipeline_components")},
    "workflow_ltx25_denoise": {("models", "pipeline_components", "denoise", "pipeline_components")},
    "workflow_ltx25_decoder": {("models", "pipeline_components", "decode", "pipeline_components")},
    "workflow_wan_animate_text_encoder": {("models", "pipeline_components", "prompt", "pipeline_components")},
    "workflow_wan_animate_image_encoder": {
        ("models", "pipeline_components", "imageEmbeddings", "pipeline_components")
    },
    "workflow_wan_animate_video_encoder": {("models", "pipeline_components", "videoEncode", "pipeline_components")},
    "workflow_wan_animate_vae_encoder": {("models", "pipeline_components", "imageEncode", "pipeline_components")},
    "workflow_wan_animate_denoise": {("models", "pipeline_components", "denoise", "pipeline_components")},
    "workflow_wan_animate_decoder": {("models", "pipeline_components", "decode", "pipeline_components")},
}
_PIPELINE_ACTION_COMPONENT_EDGES = {
    # Qwen's routed ControlNet action encodes the control image with the base
    # VAE. SDXL's ordinary ControlNet block consumes the prepared control image
    # directly, so projecting this edge onto SDXL would invent an upstream
    # component dependency that does not exist.
    ("QwenImageModularPipeline", "controlnet"): {
        ("models", "vae_out", "controlnet", "vae"),
    },
}

# These are explicit review candidates, not an automatic modelType/mode join.
# A new candidate requires a backend review of its upstream workflow, generic
# action graph, artifact, and Studio execution specification.
REVIEWED_CLUSTER_EXECUTION_CANDIDATES = (
    {
        "pipelineClass": "Flux2ModularPipeline",
        "workflowId": "text2image",
        "adapterSource": "mode",
        "adapterId": "equivalent_standard_route",
        "studioMode": "text_to_image",
        "studioSpecId": "flux2-modular:equivalent-standard-text-to-image:v1",
        "executionRoute": "equivalent_standard",
    },
    {
        "pipelineClass": "Flux2ModularPipeline",
        "workflowId": "image_conditioned",
        "adapterSource": "mode",
        "adapterId": "equivalent_standard_route",
        "studioMode": "multi_image_reference_edit",
        "studioSpecId": "flux2-modular:equivalent-standard-image-conditioned:v1",
        "executionRoute": "equivalent_standard",
    },
    {
        "pipelineClass": "FluxModularPipeline",
        "workflowId": "text2image",
        "adapterSource": "mode",
        "adapterId": "text_to_image",
        "studioMode": "text_to_image",
        "studioSpecId": "flux-dev:modular-text-to-image:v1",
    },
    {
        "pipelineClass": "FluxModularPipeline",
        "workflowId": "image2image",
        "adapterSource": "mode",
        "adapterId": "image_to_image",
        "studioMode": "image_to_image",
        "studioSpecId": "flux-dev:modular-image-to-image:v1",
    },
    {
        "pipelineClass": "FluxKontextModularPipeline",
        "workflowId": "text2image",
        "adapterSource": "mode",
        "adapterId": "text_to_image",
        "studioMode": "text_to_image",
        "studioSpecId": "flux-kontext:modular-text-to-image:v1",
    },
    {
        "pipelineClass": "FluxKontextModularPipeline",
        "workflowId": "image_conditioned",
        "adapterSource": "mode",
        "adapterId": "edit_image",
        "studioMode": "edit_image",
        "studioSpecId": "flux-kontext:modular-edit-image:v1",
    },
    {
        "pipelineClass": "Flux2KleinModularPipeline",
        "workflowId": "text2image",
        "adapterSource": "mode",
        "adapterId": "text_to_image",
        "studioMode": "text_to_image",
        "studioSpecId": "flux2-klein:modular-text-to-image:v1",
    },
    {
        "pipelineClass": "Flux2KleinModularPipeline",
        "workflowId": "image_conditioned",
        "adapterSource": "mode",
        "adapterId": "edit_image",
        "studioMode": "edit_image",
        "studioSpecId": "flux2-klein:modular-edit-image:v1",
    },
    {
        "pipelineClass": "Flux2KleinBaseModularPipeline",
        "workflowId": "text2image",
        "adapterSource": "mode",
        "adapterId": "text_to_image",
        "studioMode": "text_to_image",
        "studioSpecId": "flux2-klein-base:modular-text-to-image:v1",
    },
    {
        "pipelineClass": "Flux2KleinBaseModularPipeline",
        "workflowId": "image_conditioned",
        "adapterSource": "mode",
        "adapterId": "edit_image",
        "studioMode": "edit_image",
        "studioSpecId": "flux2-klein-base:modular-edit-image:v1",
    },
    {
        "pipelineClass": "StableDiffusionXLModularPipeline",
        "workflowId": "text2image",
        "adapterSource": "mode",
        "adapterId": "text_to_image",
        "studioMode": "text_to_image",
        "studioSpecId": "sdxl-base:modular-text-to-image:v1",
    },
    {
        "pipelineClass": "StableDiffusionXLModularPipeline",
        "workflowId": "image2image",
        "adapterSource": "mode",
        "adapterId": "image_to_image",
        "studioMode": "edit_image",
        "studioSpecId": "sdxl-base:modular-image-to-image:v1",
    },
    {
        "pipelineClass": "StableDiffusionXLModularPipeline",
        "workflowId": "inpainting",
        "adapterSource": "mode",
        "adapterId": "inpaint",
        "studioMode": "inpaint",
        "studioSpecId": "sdxl-base:modular-inpainting:v1",
    },
    {
        "pipelineClass": "StableDiffusionXLModularPipeline",
        "workflowId": "controlnet_text2image",
        "adapterSource": "mode",
        "adapterId": "control_image",
        "studioMode": "control_image",
        "studioSpecId": "sdxl-base:modular-controlnet-text-to-image:v1",
    },
    {
        "pipelineClass": "StableDiffusionXLModularPipeline",
        "workflowId": "controlnet_image2image",
        "adapterSource": "mode",
        "adapterId": "control_edit_image",
        "studioMode": "control_edit_image",
        "studioSpecId": "sdxl-base:modular-controlnet-image-to-image:v1",
    },
    {
        "pipelineClass": "StableDiffusionXLModularPipeline",
        "workflowId": "controlnet_inpainting",
        "adapterSource": "mode",
        "adapterId": "control_inpaint",
        "studioMode": "control_inpaint",
        "studioSpecId": "sdxl-base:modular-controlnet-inpainting:v1",
    },
    {
        "pipelineClass": "QwenImageEditModularPipeline",
        "workflowId": "image_conditioned",
        "adapterSource": "mode",
        "adapterId": "edit_image",
        "studioMode": "edit_image",
        "studioSpecId": "qwen-image-edit:edit-image:v1",
    },
    {
        "pipelineClass": "QwenImageEditModularPipeline",
        "workflowId": "image_conditioned_inpainting",
        "adapterSource": "state_flow",
        "adapterId": "image_conditioned_inpainting",
        "studioMode": "modular_inpainting",
        "studioSpecId": "qwen-image-edit:modular-inpainting:v1",
    },
    {
        "pipelineClass": "QwenImageEditPlusModularPipeline",
        "workflowId": "default",
        "adapterSource": "mode",
        "adapterId": "edit_image",
        "studioMode": "edit_image",
        "studioSpecId": "qwen-image-edit-plus:edit-image:v1",
    },
    {
        "pipelineClass": "QwenImageEditPlusModularPipeline",
        "workflowId": "default",
        "adapterSource": "mode",
        "adapterId": "multi_image_reference_edit",
        "studioMode": "multi_image_reference_edit",
        "studioSpecId": "qwen-image-edit-plus:multi-image-reference-edit:v1",
    },
    {
        "pipelineClass": "QwenImageLayeredModularPipeline",
        "workflowId": "default",
        "adapterSource": "mode",
        "adapterId": "layer_decomposition",
        "studioMode": "layer_decomposition",
        "studioSpecId": "qwen-image-layered:layer-decomposition:v1",
    },
    {
        "pipelineClass": "QwenImageModularPipeline",
        "workflowId": "text2image",
        "adapterSource": "mode",
        "adapterId": "text_to_image",
        "studioMode": "modular_text_to_image",
        "studioSpecId": "qwen-image-2512:modular-text-to-image:v1",
    },
    {
        "pipelineClass": "QwenImageModularPipeline",
        "workflowId": "controlnet_text2image",
        "adapterSource": "mode",
        "adapterId": "control_image",
        "studioMode": "control_image",
        "studioSpecId": "qwen-image-2512:control-image:v1",
    },
    {
        "pipelineClass": "QwenImageModularPipeline",
        "workflowId": "image2image",
        "adapterSource": "state_flow",
        "adapterId": "image2image",
        "studioMode": "image_to_image",
        "studioSpecId": "qwen-image-2512:modular-image-to-image:v1",
    },
    {
        "pipelineClass": "QwenImageModularPipeline",
        "workflowId": "inpainting",
        "adapterSource": "state_flow",
        "adapterId": "inpainting",
        "studioMode": "inpainting",
        "studioSpecId": "qwen-image-2512:modular-inpainting:v1",
    },
    {
        "pipelineClass": "QwenImageModularPipeline",
        "workflowId": "controlnet_image2image",
        "adapterSource": "state_flow",
        "adapterId": "controlnet_image2image",
        "studioMode": "control_edit_image",
        "studioSpecId": "qwen-image-2512:modular-control-image-to-image:v1",
    },
    {
        "pipelineClass": "QwenImageModularPipeline",
        "workflowId": "controlnet_inpainting",
        "adapterSource": "state_flow",
        "adapterId": "controlnet_inpainting",
        "studioMode": "control_inpaint",
        "studioSpecId": "qwen-image-2512:modular-control-inpainting:v1",
    },
    {
        "pipelineClass": "WanModularPipeline",
        "workflowId": "default",
        "adapterSource": "mode",
        "adapterId": "text_to_video",
        "studioMode": "text_to_video",
        "studioSpecId": "wan-1.3b:modular-text-to-video:v1",
    },
    {
        "pipelineClass": "WanImage2VideoModularPipeline",
        "workflowId": "image2video",
        "adapterSource": "mode",
        "adapterId": "image_to_video",
        "studioMode": "single_image_to_video",
        "studioSpecId": "wan-i2v-480p:modular-image-to-video:v1",
    },
    {
        "pipelineClass": "WanImage2VideoModularPipeline",
        "workflowId": "flf2v",
        "adapterSource": "state_flow",
        "adapterId": "flf2v",
        "studioMode": "image_to_video",
        "studioSpecId": "wan-flf:image-to-video:v1",
    },
    {
        "pipelineClass": "ZImageModularPipeline",
        "workflowId": "text2image",
        "adapterSource": "mode",
        "adapterId": "text_to_image",
        "studioMode": "modular_text_to_image",
        "studioSpecId": "z-image:modular-text-to-image:v1",
    },
    {
        "pipelineClass": "ZImageModularPipeline",
        "workflowId": "image2image",
        "adapterSource": "mode",
        "adapterId": "image_to_image",
        "studioMode": "modular_image_to_image",
        "studioSpecId": "z-image:modular-image-to-image:v1",
    },
    {
        "pipelineClass": "AnimaModularPipeline",
        "workflowId": "text2image",
        "adapterSource": "workflow",
        "adapterId": "official_top_level_blocks",
        "studioMode": "text_to_image",
        "studioSpecId": "anima:modular-text-to-image:v1",
        "executionRoute": "official_modular_workflow",
    },
    {
        "pipelineClass": "AnimaModularPipeline",
        "workflowId": "img2img",
        "adapterSource": "workflow",
        "adapterId": "official_top_level_blocks",
        "studioMode": "image_to_image",
        "studioSpecId": "anima:modular-image-to-image:v1",
        "executionRoute": "official_modular_workflow",
    },
    {
        "pipelineClass": "MiniMaxMusic3ModularPipeline",
        "workflowId": "default",
        "adapterSource": "workflow",
        "adapterId": "official_top_level_blocks",
        "studioMode": "text_to_audio",
        "studioSpecId": "minimax-music3:modular-text-to-audio:v1",
        "executionRoute": "official_modular_workflow",
    },
)

REVIEWED_CLUSTER_EXECUTION_CANDIDATES += tuple(
    {
        "pipelineClass": pipeline_class,
        "workflowId": workflow_id,
        "adapterSource": "workflow",
        "adapterId": "official_top_level_blocks",
        "studioMode": studio_mode,
        "studioSpecId": f"{slug}:modular-{studio_mode.replace('_', '-')}:v1",
        "executionRoute": "official_modular_workflow",
    }
    for pipeline_class, slug in (
        ("HeliosModularPipeline", "helios-base"),
        ("HeliosPyramidModularPipeline", "helios-pyramid"),
        ("HeliosPyramidDistilledModularPipeline", "helios-pyramid-distilled"),
    )
    for workflow_id, studio_mode in (
        ("text2video", "text_to_video"),
        ("image2video", "image_to_video"),
        ("video2video", "video_to_video"),
    )
)

REVIEWED_CLUSTER_EXECUTION_CANDIDATES += (
    {
        "pipelineClass": "HunyuanVideo15ModularPipeline",
        "workflowId": "text2video",
        "adapterSource": "workflow",
        "adapterId": "official_top_level_blocks",
        "studioMode": "text_to_video",
        "studioSpecId": "hunyuan-video15:modular-text-to-video:v1",
        "executionRoute": "official_modular_workflow",
    },
    {
        "pipelineClass": "HunyuanVideo15ModularPipeline",
        "workflowId": "image2video",
        "adapterSource": "workflow",
        "adapterId": "official_top_level_blocks",
        "studioMode": "image_to_video",
        "studioSpecId": "hunyuan-video15:modular-image-to-video:v1",
        "executionRoute": "official_modular_workflow",
    },
)

REVIEWED_CLUSTER_EXECUTION_CANDIDATES += (
    {
        "pipelineClass": "Cosmos3DistilledModularPipeline",
        "workflowId": "text2image",
        "adapterSource": "workflow",
        "adapterId": "official_top_level_blocks",
        "studioMode": "text_to_image",
        "studioSpecId": "cosmos3-distilled:modular-text-to-image:v1",
        "executionRoute": "official_modular_workflow",
    },
    {
        "pipelineClass": "Cosmos3DistilledModularPipeline",
        "workflowId": "image2video",
        "adapterSource": "workflow",
        "adapterId": "official_top_level_blocks",
        "studioMode": "image_to_video",
        "studioSpecId": "cosmos3-distilled:modular-image-to-video:v1",
        "executionRoute": "official_modular_workflow",
    },
)

REVIEWED_CLUSTER_EXECUTION_CANDIDATES += (
    {
        "pipelineClass": "MiniMaxH3ModularPipeline",
        "workflowId": "t2va",
        "adapterSource": "workflow",
        "adapterId": "official_top_level_blocks",
        "studioMode": "text_to_video_with_audio",
        "studioSpecId": "minimax-h3:modular-text-to-video-with-audio:v1",
        "executionRoute": "official_modular_workflow",
    },
    {
        "pipelineClass": "MiniMaxH3ModularPipeline",
        "workflowId": "fl2va",
        "adapterSource": "workflow",
        "adapterId": "official_top_level_blocks",
        "studioMode": "first_last_frame_to_video_with_audio",
        "studioSpecId": "minimax-h3:modular-first-last-frame-to-video-with-audio:v1",
        "executionRoute": "official_modular_workflow",
    },
    {
        "pipelineClass": "MiniMaxH3ModularPipeline",
        "workflowId": "ref2va",
        "adapterSource": "workflow",
        "adapterId": "official_top_level_blocks",
        "studioMode": "reference_to_video_with_audio",
        "studioSpecId": "minimax-h3:modular-reference-to-video-with-audio:v1",
        "executionRoute": "official_modular_workflow",
    },
)

REVIEWED_CLUSTER_EXECUTION_CANDIDATES += (
    {
        "pipelineClass": "Cosmos3OmniModularPipeline",
        "workflowId": "text2image",
        "adapterSource": "workflow",
        "adapterId": "official_top_level_blocks",
        "studioMode": "text_to_image",
        "studioSpecId": "cosmos3-nano:modular-text-to-image:v1",
        "executionRoute": "official_modular_workflow",
    },
    {
        "pipelineClass": "Cosmos3OmniModularPipeline",
        "workflowId": "text2video",
        "adapterSource": "workflow",
        "adapterId": "official_top_level_blocks",
        "studioMode": "text_to_video",
        "studioSpecId": "cosmos3-nano:modular-text-to-video:v1",
        "executionRoute": "official_modular_workflow",
    },
    {
        "pipelineClass": "Cosmos3OmniModularPipeline",
        "workflowId": "image2video",
        "adapterSource": "workflow",
        "adapterId": "official_top_level_blocks",
        "studioMode": "image_to_video",
        "studioSpecId": "cosmos3-nano:modular-image-to-video:v1",
        "executionRoute": "official_modular_workflow",
    },
    {
        "pipelineClass": "Cosmos3OmniModularPipeline",
        "workflowId": "video2video",
        "adapterSource": "workflow",
        "adapterId": "official_top_level_blocks",
        "studioMode": "video_to_video",
        "studioSpecId": "cosmos3-nano:modular-video-to-video:v1",
        "executionRoute": "official_modular_workflow",
    },
    {
        "pipelineClass": "Cosmos3OmniModularPipeline",
        "workflowId": "text2video_with_sound",
        "adapterSource": "workflow",
        "adapterId": "official_top_level_blocks",
        "studioMode": "text_to_video_with_audio",
        "studioSpecId": "cosmos3-nano:modular-text-to-video-with-audio:v1",
        "executionRoute": "official_modular_workflow",
    },
    {
        "pipelineClass": "Cosmos3OmniModularPipeline",
        "workflowId": "image2video_with_sound",
        "adapterSource": "workflow",
        "adapterId": "official_top_level_blocks",
        "studioMode": "image_to_video_with_audio",
        "studioSpecId": "cosmos3-nano:modular-image-to-video-with-audio:v1",
        "executionRoute": "official_modular_workflow",
    },
    {
        "pipelineClass": "Cosmos3OmniModularPipeline",
        "workflowId": "video2video_with_sound",
        "adapterSource": "workflow",
        "adapterId": "official_top_level_blocks",
        "studioMode": "video_to_video_with_audio",
        "studioSpecId": "cosmos3-nano:modular-video-to-video-with-audio:v1",
        "executionRoute": "official_modular_workflow",
    },
)

REVIEWED_CLUSTER_EXECUTION_CANDIDATES += tuple(
    {
        "pipelineClass": pipeline_class,
        "workflowId": "default",
        "adapterSource": "workflow",
        "adapterId": "official_top_level_blocks",
        "studioMode": "character_animate",
        "studioSpecId": f"{slug}:modular-character-animate:v1",
        "executionRoute": "official_modular_workflow",
    }
    for pipeline_class, slug in (
        ("WanAnimate2ModularPipeline", "wan-animate-2"),
        ("WanAnimate2DistilledModularPipeline", "wan-animate-2-distilled"),
    )
)

REVIEWED_CLUSTER_EXECUTION_CANDIDATES += tuple(
    {
        "pipelineClass": "StableDiffusionXLModularPipeline",
        "workflowId": workflow_id,
        "adapterSource": "state_flow",
        "adapterId": workflow_id,
        "studioMode": studio_mode,
        "studioSpecId": f"sdxl-base:modular-{spec_name}:v1",
    }
    for workflow_id, studio_mode, spec_name in (
        ("controlnet_union_text2image", "control_union_image", "controlnet-union-text-to-image"),
        ("controlnet_union_image2image", "control_union_edit_image", "controlnet-union-image-to-image"),
        ("controlnet_union_inpainting", "control_union_inpaint", "controlnet-union-inpainting"),
        ("ip_adapter_text2image", "ip_adapter_image", "ip-adapter-text-to-image"),
        ("ip_adapter_image2image", "ip_adapter_edit_image", "ip-adapter-image-to-image"),
        ("ip_adapter_inpainting", "ip_adapter_inpaint", "ip-adapter-inpainting"),
        (
            "ip_adapter_controlnet_text2image",
            "ip_adapter_control_image",
            "ip-adapter-controlnet-text-to-image",
        ),
        (
            "ip_adapter_controlnet_image2image",
            "ip_adapter_control_edit_image",
            "ip-adapter-controlnet-image-to-image",
        ),
        (
            "ip_adapter_controlnet_inpainting",
            "ip_adapter_control_inpaint",
            "ip-adapter-controlnet-inpainting",
        ),
        (
            "ip_adapter_controlnet_union_text2image",
            "ip_adapter_control_union_image",
            "ip-adapter-controlnet-union-text-to-image",
        ),
        (
            "ip_adapter_controlnet_union_image2image",
            "ip_adapter_control_union_edit_image",
            "ip-adapter-controlnet-union-image-to-image",
        ),
        (
            "ip_adapter_controlnet_union_inpainting",
            "ip_adapter_control_union_inpaint",
            "ip-adapter-controlnet-union-inpainting",
        ),
    )
)

REVIEWED_CLUSTER_EXECUTION_CANDIDATES += (
    {
        "pipelineClass": "ErnieImageModularPipeline",
        "workflowId": "text2image",
        "adapterSource": "mode",
        "adapterId": "equivalent_standard_route",
        "studioMode": "text_to_image",
        "studioSpecId": "ernie-image:equivalent-standard-text-to-image:v1",
        "executionRoute": "equivalent_standard",
    },
    {
        "pipelineClass": "LTXModularPipeline",
        "workflowId": "text2video",
        "adapterSource": "mode",
        "adapterId": "equivalent_standard_route",
        "studioMode": "text_to_video",
        "studioSpecId": "ltx:equivalent-standard-text-to-video:v1",
        "executionRoute": "equivalent_standard",
    },
    {
        "pipelineClass": "LTXModularPipeline",
        "workflowId": "image2video",
        "adapterSource": "mode",
        "adapterId": "equivalent_standard_route",
        "studioMode": "image_to_video",
        "studioSpecId": "ltx:equivalent-standard-image-to-video:v1",
        "executionRoute": "equivalent_standard",
    },
    {
        "pipelineClass": "Wan22ModularPipeline",
        "workflowId": "default",
        "adapterSource": "mode",
        "adapterId": "equivalent_standard_route",
        "studioMode": "text_to_video",
        "studioSpecId": "wan22:equivalent-standard-text-to-video:v1",
        "executionRoute": "equivalent_standard",
    },
    {
        "pipelineClass": "Wan22Image2VideoModularPipeline",
        "workflowId": "default",
        "adapterSource": "mode",
        "adapterId": "equivalent_standard_route",
        "studioMode": "image_to_video",
        "studioSpecId": "wan22-i2v:equivalent-standard-image-to-video:v1",
        "executionRoute": "equivalent_standard",
    },
    {
        "pipelineClass": "LTX2ModularPipeline",
        "workflowId": "text2video",
        "adapterSource": "mode",
        "adapterId": "equivalent_standard_route",
        "studioMode": "text_to_video",
        "studioSpecId": "ltx2-modular:equivalent-standard-text-to-video:v1",
        "executionRoute": "equivalent_standard",
    },
    {
        "pipelineClass": "LTX2ModularPipeline",
        "workflowId": "image2video",
        "adapterSource": "mode",
        "adapterId": "equivalent_standard_route",
        "studioMode": "image_to_video",
        "studioSpecId": "ltx2-modular:equivalent-standard-image-to-video:v1",
        "executionRoute": "equivalent_standard",
    },
    {
        "pipelineClass": "LTX2ModularPipeline",
        "workflowId": "condition",
        "adapterSource": "mode",
        "adapterId": "equivalent_standard_route",
        "studioMode": "reference_to_video",
        "studioSpecId": "ltx2-modular:equivalent-standard-condition-images:v1",
        "executionRoute": "equivalent_standard",
    },
    {
        "pipelineClass": "LTX2ModularPipeline",
        "workflowId": "in_context",
        "adapterSource": "mode",
        "adapterId": "equivalent_standard_route",
        "studioMode": "in_context_to_video",
        "studioSpecId": "ltx2-modular:equivalent-standard-in-context-canny:v1",
        "executionRoute": "equivalent_standard",
    },
)

_WORKFLOW_REPOSITORIES = {
    "Cosmos3DistilledModularPipeline": frozenset(
        {
            ("text2image", "nvidia/Cosmos3-Super-Text2Image-4Step"),
            ("image2video", "nvidia/Cosmos3-Super-Image2Video-4Step"),
        }
    ),
    "MiniMaxH3ModularPipeline": frozenset(
        {
            ("t2va", "MiniMaxAI/MiniMax-H3"),
            ("fl2va", "MiniMaxAI/MiniMax-H3"),
            ("ref2va", "MiniMaxAI/MiniMax-H3"),
        }
    ),
    "HunyuanVideo15ModularPipeline": frozenset(
        {
            (
                "text2video",
                "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
            ),
            (
                "image2video",
                "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled",
            ),
        }
    ),
    "WanModularPipeline": frozenset(
        {
            ("default", WAN_T2V_REPOSITORY),
            ("default", WAN_T2V_14B_REPOSITORY),
        }
    ),
    "WanImage2VideoModularPipeline": frozenset(WAN_WORKFLOW_REPOSITORIES),
}


def _reason(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _role_map(definition: Mapping[str, Any]) -> dict[str, str]:
    return {str(role): str(node_key) for role, node_key, _x, _y in definition.get("roles", ())}


def _edge_set(definition: Mapping[str, Any]) -> set[tuple[str, str, str, str]]:
    return {tuple(str(value) for value in edge) for edge in definition.get("edges", ())}


def _binding_set(definition: Mapping[str, Any]) -> set[tuple[str, str, str]]:
    return {tuple(str(value) for value in binding) for binding in definition.get("bindings", ())}


def _audit_candidate(
    candidate: Mapping[str, str],
    *,
    definitions: Mapping[tuple[str, str], Mapping[str, Any]],
    studio_definitions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    pipeline_class = candidate["pipelineClass"]
    workflow_id = candidate["workflowId"]
    adapter_source = candidate["adapterSource"]
    adapter_id = candidate["adapterId"]
    studio_mode = candidate["studioMode"]
    studio_spec_id = candidate["studioSpecId"]
    equivalent_standard_route = candidate.get("executionRoute") == "equivalent_standard"
    official_modular_workflow_route = candidate.get("executionRoute") == "official_modular_workflow"
    definition = definitions.get((pipeline_class, workflow_id))
    studio_definition = studio_definitions.get(studio_spec_id)
    reasons: list[dict[str, str]] = []
    adapter: Mapping[str, Any] | None = None

    if definition is None:
        reasons.append(_reason("unknown_cluster_definition", "The exact pipeline/workflow definition is absent."))
    else:
        expected_integration_status = (
            "equivalent_standard_route"
            if equivalent_standard_route
            else "reviewed_modular_workflow_route"
            if official_modular_workflow_route
            else "reviewed_modiff_contract"
        )
        if definition.get("integrationStatus") != expected_integration_status:
            reasons.append(
                _reason(
                    "integration_route_mismatch",
                    "The candidate execution route differs from the reviewed library integration status.",
                )
            )
        matches = [
            item
            for item in definition["graphAdapterContracts"]
            if item["source"] == adapter_source and item["adapterId"] == adapter_id
        ]
        if len(matches) != 1:
            reasons.append(
                _reason(
                    "unknown_graph_adapter",
                    "The exact workflow has no unique reviewed graph-adapter contract for this candidate.",
                )
            )
        else:
            adapter = matches[0]
            if adapter["upstreamWorkflowId"] != workflow_id:
                reasons.append(
                    _reason("workflow_identity_mismatch", "The graph adapter selects a different upstream workflow.")
                )

    public_spec = None
    if studio_definition is None:
        reasons.append(_reason("unknown_studio_spec", "The declared Studio execution specification is absent."))
    else:
        public_spec = studio_execution_spec_for_pair(
            str(studio_definition.get("modelType") or ""),
            str(studio_definition.get("mode") or ""),
        )
        if public_spec is None or public_spec.get("id") != studio_spec_id:
            reasons.append(
                _reason("ambiguous_studio_spec", "The model-type/mode pair does not resolve to this unique spec.")
            )

        profile = studio_definition.get("profile")
        if not isinstance(profile, Mapping):
            reasons.append(_reason("invalid_execution_profile", "The Studio execution profile is malformed."))
        else:
            if studio_definition.get("modelType") != pipeline_class or studio_definition.get("mode") != studio_mode:
                reasons.append(
                    _reason("studio_identity_mismatch", "The Studio model type or mode differs from the adapter.")
                )
            if equivalent_standard_route:
                execution_path = str(profile.get("execution_path") or "")
                loader_contract = _DIRECT_LOADER_CONTRACTS.get(execution_path)
                expected_execution_class = _equivalent_standard_execution_class(pipeline_class, workflow_id)
                upstream_targets = equivalent_modular_targets(pipeline_class, workflow_id)
                resolved_upstream_class = (
                    "WanPipeline" if expected_execution_class == "Wan22Pipeline" else expected_execution_class
                )
                if loader_contract is None:
                    reasons.append(
                        _reason(
                            "execution_path_not_reviewed_equivalent",
                            "The equivalent route does not use a reviewed standard Diffusers execution path.",
                        )
                    )
                elif (profile.get("loader_module"), profile.get("loader_action")) != loader_contract:
                    reasons.append(
                        _reason(
                            "loader_contract_mismatch",
                            "The equivalent route does not use the exact reviewed full-pipeline loader.",
                        )
                    )
                if (
                    profile.get("model_type") != pipeline_class
                    or profile.get("pipeline_class") != expected_execution_class
                    or resolved_upstream_class not in upstream_targets
                ):
                    reasons.append(
                        _reason(
                            "equivalent_pipeline_class_mismatch",
                            "The execution profile does not target the exact reviewed standard equivalent.",
                        )
                    )
            else:
                if profile.get("execution_path") != "modular-diffusers":
                    reasons.append(
                        _reason(
                            "execution_path_not_modular",
                            "The Studio spec uses a standard/direct pipeline rather than Modular Diffusers actions.",
                        )
                    )
                if (
                    profile.get("loader_module") != "modules.ModularDiffusers"
                    or profile.get("loader_action") != "ModelsLoader"
                ):
                    reasons.append(
                        _reason("loader_contract_mismatch", "The Studio spec does not use the reviewed ModelsLoader.")
                    )
                if profile.get("model_type") != pipeline_class or profile.get("pipeline_class") != pipeline_class:
                    reasons.append(
                        _reason("pipeline_class_mismatch", "The execution profile targets a different pipeline class.")
                    )
            if studio_mode not in profile.get("modes", ()):
                reasons.append(_reason("profile_mode_mismatch", "The execution profile does not declare this mode."))

            repository = profile.get("default_repo")
            if not isinstance(repository, str) or not repository:
                reasons.append(_reason("missing_reviewed_artifact", "The execution profile has no default artifact."))
            else:
                try:
                    artifact_revision = (
                        require_catalog_revision(repository)
                        if equivalent_standard_route
                        else require_catalog_revision(repository, model_type=pipeline_class)
                    )
                except ValueError as error:
                    artifact_revision = None
                    reasons.append(_reason("unresolved_artifact_revision", str(error)))
                workflow_repositories = _WORKFLOW_REPOSITORIES.get(pipeline_class)
                if workflow_repositories is not None and (workflow_id, repository) not in workflow_repositories:
                    reasons.append(
                        _reason(
                            "artifact_workflow_mismatch",
                            "The reviewed artifact belongs to a different upstream workflow for this pipeline.",
                        )
                    )

        roles = _role_map(studio_definition)
        edges = _edge_set(studio_definition)
        bindings = _binding_set(studio_definition)
        if equivalent_standard_route:
            profile = studio_definition.get("profile")
            loader_node_key = (
                f"{profile.get('loader_module')}.{profile.get('loader_action')}"
                if isinstance(profile, Mapping)
                else ""
            )
            if list(roles.values()).count(loader_node_key) != 1:
                reasons.append(
                    _reason(
                        "missing_equivalent_pipeline_loader",
                        "The Studio graph does not contain one exact reviewed equivalent full-pipeline loader.",
                    )
                )
            required_loader_bindings = {
                ("pipeline_class", "pipelineClass"),
                ("execution_profile_id", "executionProfileId"),
                ("model_id", "artifact"),
            }
            actual_loader_bindings = {
                (field, source) for role, field, source in bindings if roles.get(role) == loader_node_key
            }
            if not required_loader_bindings.issubset(actual_loader_bindings):
                reasons.append(
                    _reason(
                        "missing_loader_identity_bindings",
                        "The equivalent full-pipeline graph does not seal its loader identity.",
                    )
                )
        else:
            if roles.get("models") != _MODELS_LOADER:
                reasons.append(
                    _reason("missing_models_loader_role", "The Studio graph has no exact ModelsLoader role.")
                )
            missing_loader_bindings = sorted(_REQUIRED_LOADER_BINDINGS - bindings)
            if missing_loader_bindings:
                reasons.append(
                    _reason("missing_loader_identity_bindings", "The Studio graph does not seal its loader identity.")
                )

        if adapter is not None:
            if equivalent_standard_route:
                if adapter["actionSequence"] != ["full_pipeline"] or adapter["stateEdges"]:
                    reasons.append(
                        _reason(
                            "equivalent_adapter_mismatch",
                            "The equivalent route must declare exactly one full-pipeline action and no split state edges.",
                        )
                    )
            else:
                expected_action_roles: dict[str, str] = {}
                for action in adapter["actionSequence"]:
                    role_contract = _ACTION_ROLE_CONTRACTS.get(action)
                    if role_contract is None:
                        reasons.append(
                            _reason(
                                "unknown_action_contract", f"No generic Studio role contract exists for {action!r}."
                            )
                        )
                        continue
                    role, node_key = role_contract
                    expected_action_roles[action] = role
                    if roles.get(role) != node_key:
                        reasons.append(
                            _reason(
                                "action_role_mismatch",
                                f"Action {action!r} is not materialized by the exact reviewed Studio node.",
                            )
                        )
                    required_component_edges = _REQUIRED_COMPONENT_EDGES[
                        action
                    ] | _PIPELINE_ACTION_COMPONENT_EDGES.get((pipeline_class, action), set())
                    missing_component_edges = required_component_edges - edges
                    if missing_component_edges:
                        reasons.append(
                            _reason(
                                "missing_component_binding",
                                f"Action {action!r} is missing a required model/component edge.",
                            )
                        )

                for state_edge in adapter["stateEdges"]:
                    producer_role = expected_action_roles.get(state_edge["producerAction"])
                    consumer_role = expected_action_roles.get(state_edge["consumerAction"])
                    projected = (
                        producer_role,
                        state_edge["producerOutput"],
                        consumer_role,
                        state_edge["consumerInput"],
                    )
                    if producer_role is None or consumer_role is None or projected not in edges:
                        reasons.append(
                            _reason(
                                "state_edge_mismatch",
                                "A required upstream PipelineState transfer is absent from the Studio graph.",
                            )
                        )

                action_node_keys = {
                    _ACTION_ROLE_CONTRACTS[action][1]
                    for action in adapter["actionSequence"]
                    if action in _ACTION_ROLE_CONTRACTS
                }
                unexpected_nodes = sorted(set(roles.values()) - action_node_keys - _AUXILIARY_NODE_KEYS)
                if unexpected_nodes:
                    reasons.append(
                        _reason(
                            "unexpected_graph_action",
                            "The Studio graph contains an executable node outside the reviewed action/utility set.",
                        )
                    )

            bound_upstream_inputs = {
                _UPSTREAM_INPUT_BY_BINDING_SOURCE[source]
                for _role, _param, source in bindings
                if source in _UPSTREAM_INPUT_BY_BINDING_SOURCE
            }
            bound_upstream_inputs.update(
                upstream_input
                for (route_pipeline, source), upstream_input in _PIPELINE_UPSTREAM_INPUT_BY_BINDING_SOURCE.items()
                if route_pipeline == pipeline_class and any(binding[2] == source for binding in bindings)
            )
            required_upstream_inputs = set(adapter["requiredInputs"])
            expected_bound_inputs = required_upstream_inputs | set(
                _OPTIONAL_UPSTREAM_INPUTS.get((pipeline_class, workflow_id), ())
            )
            if bound_upstream_inputs != expected_bound_inputs:
                reasons.append(
                    _reason(
                        "required_upstream_inputs_mismatch",
                        "The Studio graph's required user media/prompt inputs differ from the upstream workflow.",
                    )
                )

    binding_sources = sorted(
        {str(source) for _role, _param, source in (studio_definition.get("bindings", ()) if studio_definition else ())}
    )
    definition_inputs = {
        str(field.get("name"))
        for field in (definition.get("inputs", ()) if definition else ())
        if isinstance(field, Mapping)
    }
    instance_input_bindings = [
        {"bindingSource": source, "input": input_name}
        for source, input_name in sorted(_INSTANCE_INPUT_BY_BINDING_SOURCE.items())
        if source in binding_sources and input_name in definition_inputs
    ]
    instance_binding_sources = {item["bindingSource"] for item in instance_input_bindings}
    referenced_fields = {
        (str(role), str(field))
        for role, field, _source in (studio_definition.get("bindings", ()) if studio_definition else ())
    }
    referenced_fields.update(
        (str(role), str(field))
        for source_role, source_field, target_role, target_field in (
            studio_definition.get("edges", ()) if studio_definition else ()
        )
        for role, field in ((source_role, source_field), (target_role, target_field))
    )
    dynamic_field_actions = [
        {"role": role, "field": field, "event": event, "valueSource": value_source}
        for role, field, event, value_source in _DYNAMIC_FIELD_ACTIONS
        if (role, field) in referenced_fields and value_source in binding_sources
    ]
    artifact = (
        {
            "repo": studio_definition["profile"]["default_repo"],
            "revision": artifact_revision,
        }
        if studio_definition is not None
        and isinstance(studio_definition.get("profile"), Mapping)
        and isinstance(studio_definition["profile"].get("default_repo"), str)
        and studio_definition["profile"].get("default_repo")
        else None
    )
    model_dependencies = studio_model_dependencies_for_pair(pipeline_class, studio_mode)
    dependency_sources = set(binding_sources) & {"kind", "repo", "revision"}
    controlnet_dependency = next(
        (dependency for dependency in model_dependencies if dependency.get("kind") == "controlnet"),
        None,
    )
    ip_adapter_dependency = next(
        (dependency for dependency in model_dependencies if dependency.get("id") == "sdxl-ip-adapter"),
        None,
    )
    ip_adapter_pin = (
        catalog_repository_pin(ip_adapter_dependency["repo"]) if isinstance(ip_adapter_dependency, Mapping) else None
    )
    adapter_dependency = next(
        (dependency for dependency in model_dependencies if dependency.get("kind") == "adapter"),
        None,
    )
    adapter_pin = (
        catalog_repository_pin(adapter_dependency["repo"]) if isinstance(adapter_dependency, Mapping) else None
    )
    union_controlnet = "control_union" in studio_mode
    execution_profile = studio_definition.get("profile") if isinstance(studio_definition, Mapping) else None
    execution_pipeline_class = (
        execution_profile.get("pipeline_class")
        if equivalent_standard_route and isinstance(execution_profile, Mapping)
        else pipeline_class
    )
    sealed_binding_values: dict[str, Any] = {
        source: value
        for source, value in {
            "artifact": artifact["repo"] if artifact else None,
            "pipelineClass": execution_pipeline_class,
            "executionProfileId": public_spec.get("executionProfileId") if public_spec else None,
            "defaultRevision": artifact["revision"] if artifact else None,
            "mode": studio_mode,
            "empty": "",
            "true": True,
            "false": False,
            "addAlpha": "add alpha",
            "removeAlpha": "remove alpha",
            "ordinary": "ordinary",
            "fp16": "fp16",
            "controlnetKind": controlnet_dependency.get("kind") if controlnet_dependency else None,
            "controlnetRepo": controlnet_dependency.get("repo") if controlnet_dependency else None,
            "controlnetRevision": controlnet_dependency.get("revision") if controlnet_dependency else None,
            "controlnetWeightVariant": "" if union_controlnet else "fp16",
            "controlnetRouteVariant": "union" if union_controlnet else "ordinary",
            "controlnetLoadClass": "ControlNetUnionModel" if union_controlnet else "",
            "ipAdapterRepo": ip_adapter_dependency.get("repo") if ip_adapter_dependency else None,
            "ipAdapterRevision": ip_adapter_dependency.get("revision") if ip_adapter_dependency else None,
            "ipAdapterWeightName": ip_adapter_pin.get("weightName") if isinstance(ip_adapter_pin, Mapping) else None,
            "adapterWeightName": adapter_pin.get("weightName") if isinstance(adapter_pin, Mapping) else None,
            "classifierFreeGuidance": "ClassifierFreeGuidance",
            "defaultWorkflow": "default",
            "workflowId": workflow_id,
            "workflowTextEncoderBlock": "text_encoder",
            "workflowBeforeEncodeBlock": "before_encode",
            "workflowImageEncoderBlock": "vae_encoder",
            "workflowVaeEncoderBlock": "vae_encoder",
            "workflowImageEmbeddingsBlock": "image_encoder",
            "semanticGeneratorBlock": "semantic_generator",
            "workflowDenoiseBlock": "denoise",
            "workflowDecodeBlock": "decode",
            "workflowAfterDecodeBlock": "after_decode",
            "oneFrame": 1,
            "oneVideo": 1,
            "sampleRate44100": 44100,
            "distilledSteps4": 4,
            "distilledGuidance1": 1,
        }.items()
        if source in binding_sources and value is not None
    }
    if dependency_sources:
        if len(model_dependencies) != 1:
            reasons.append(
                _reason(
                    "dependency_binding_ambiguous",
                    "The Studio graph's sealed component binding does not resolve to one exact model dependency.",
                )
            )
        else:
            dependency = model_dependencies[0]
            sealed_binding_values.update(
                {source: dependency[source] for source in sorted(dependency_sources) if source in dependency}
            )
    missing_sealed_sources = sorted(
        source for source in set(binding_sources) & _SEALED_BINDING_SOURCES if source not in sealed_binding_values
    )
    if missing_sealed_sources:
        reasons.append(
            _reason(
                "sealed_binding_unresolved",
                "The Studio graph contains a sealed binding source without an exact backend value.",
            )
        )
    status = "admitted" if not reasons else "rejected"
    admission_id = f"diffusers.cluster-admission:{pipeline_class}:{workflow_id}:{adapter_source}:{adapter_id}"
    definition_id = (
        definition["id"] if definition is not None else f"diffusers.modular:{pipeline_class}:{workflow_id}"
    )
    studio_execution_spec = (
        {
            "id": public_spec["id"],
            "contentHash": public_spec["contentHash"],
            "executionProfileId": public_spec["executionProfileId"],
        }
        if public_spec is not None
        else None
    )
    profile = studio_definition.get("profile") if isinstance(studio_definition, Mapping) else None
    promotion_receipt = (
        promotion_receipt_for_admission(
            admission_id,
            definition_id=definition_id,
            artifact=artifact,
            studio_execution_spec=studio_execution_spec,
        )
        if status == "admitted" and artifact is not None and studio_execution_spec is not None
        else None
    )
    live_proof = bool(profile.get("live_proof")) if isinstance(profile, Mapping) else False
    live_proof = live_proof or promotion_receipt is not None
    publication_reasons = (
        [
            _reason(
                "runtime_resource_admission_required",
                "The materialized Cluster graph must pass the exact optional-runtime and resource checks before Run is enabled.",
            ),
            *(
                []
                if live_proof
                else [
                    _reason(
                        "live_output_review_pending",
                        "The exact Cluster execution route has not received approved visible-frontend live-output evidence.",
                    )
                ]
            ),
        ]
        if status == "admitted"
        else [
            _reason(
                "static_admission_rejected",
                "The definition remains catalog-only because its exact graph contract was rejected.",
            )
        ]
    )
    return {
        "schemaVersion": CLUSTER_EXECUTION_ADMISSION_SCHEMA_VERSION,
        "id": admission_id,
        "definitionId": definition_id,
        "studioMode": studio_mode,
        "bindingSources": binding_sources,
        "instanceInputBindings": instance_input_bindings,
        "executionParameterSources": sorted(set(binding_sources) - instance_binding_sources - _SEALED_BINDING_SOURCES),
        "sealedBindingValues": sealed_binding_values,
        "modelDependencies": model_dependencies,
        "dynamicFieldActions": dynamic_field_actions,
        "adapterContractId": adapter["id"] if adapter is not None else None,
        "studioExecutionSpec": studio_execution_spec,
        "artifact": artifact,
        "status": status,
        "claim": "static_graph_contract_compatible" if status == "admitted" else "rejected",
        "executable": False,
        "publication": {
            "schemaVersion": 1,
            "readiness": "graph_qualified" if status == "admitted" else "catalog_only",
            "insertable": status == "admitted",
            "executable": False,
            "autoEligible": False,
            "liveProof": live_proof,
            "reasons": publication_reasons,
        },
        "reasons": reasons,
    }


def audit_reviewed_cluster_execution_candidates(
    library: Mapping[str, Any] | None = None,
    *,
    studio_definitions: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return detached, deterministic static admission results.

    ``executable`` deliberately remains false after a successful audit.  A
    later publication layer must also materialize the graph, join resource and
    optional-runtime contracts, and record the appropriate runtime evidence.
    """

    if library is None:
        # Keep the node-library import lazy so its builder can attach admission
        # results without creating a module-import cycle.
        from modiff.huggingface_node_library import build_huggingface_node_library

        selected_library = build_huggingface_node_library()
    else:
        selected_library = library
    definitions = {
        (definition["pipelineClass"], definition["workflowId"]): definition
        for definition in selected_library["definitions"]
    }
    selected_studio_definitions = (
        studio_definitions if studio_definitions is not None else STUDIO_EXECUTION_SPEC_DEFINITIONS
    )
    results = [
        _audit_candidate(
            candidate,
            definitions=definitions,
            studio_definitions=selected_studio_definitions,
        )
        for candidate in REVIEWED_CLUSTER_EXECUTION_CANDIDATES
    ]
    return deepcopy(results)
