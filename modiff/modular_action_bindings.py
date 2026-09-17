"""Existing Modular actions shared by graph admission and operation discovery.

These identities select the existing executors; this is not another graph or
runtime registry. Keep saved action names stable.
"""

MODULAR_ACTION_BINDINGS = {
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


# Existing typed media helpers are offered where a reviewed stage consumes their
# output. They are ordinary nodes, not additional upstream stages or state edges.
MODULAR_AUXILIARY_OPERATION_BINDINGS = {
    "minimax_h3_references": (
        "diffusion.assemble_references",
        "modules.ModularDiffusers.WorkflowMiniMaxH3ReferenceAssembler",
    ),
}
