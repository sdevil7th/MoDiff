"""Pinned Modular Diffusers workflow truth used by MoDiff contract tests.

This module is intentionally data-only: importing capability metadata must not
instantiate Diffusers pipelines or import the model stack.  The corresponding
tests instantiate the no-weight block definitions from the reviewed Diffusers
revision and compare them with this matrix.

The matrix distinguishes upstream availability from MoDiff executability.  An
upstream workflow is not a runnable MoDiff mode until the listed generic node
actions can carry every required user input and intermediate state edge.
"""

from __future__ import annotations

from dataclasses import dataclass


PINNED_DIFFUSERS_REVISION = "2f7e0154a9db246e95c9ede43edba7db5b130805"
WAN_T2V_REPOSITORY = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN_T2V_14B_REPOSITORY = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
WAN_I2V_REPOSITORY = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
WAN_I2V_720P_REPOSITORY = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
WAN_FLF_REPOSITORY = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"
WAN_ANIMATE_2_REPOSITORY = "Wan-AI/Wan2.2-Animate-2-14B-Diffusers"
WAN_ANIMATE_2_DISTILLED_REPOSITORY = "Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers"
FLUX_DEV_REPOSITORY = "black-forest-labs/FLUX.1-dev"
FLUX_KONTEXT_REPOSITORY = "black-forest-labs/FLUX.1-Kontext-dev"
FLUX2_KLEIN_REPOSITORY = "black-forest-labs/FLUX.2-klein-4B"
FLUX2_KLEIN_BASE_REPOSITORY = "black-forest-labs/FLUX.2-klein-base-4B"
QWEN_IMAGE_REPOSITORY = "Qwen/Qwen-Image"
QWEN_IMAGE_2512_REPOSITORY = "Qwen/Qwen-Image-2512"
HUNYUAN_VIDEO_15_T2V_REPOSITORY = (
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
)
HUNYUAN_VIDEO_15_I2V_REPOSITORY = (
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled"
)
COSMOS3_DISTILLED_T2I_REPOSITORY = "nvidia/Cosmos3-Super-Text2Image-4Step"
COSMOS3_DISTILLED_I2V_REPOSITORY = "nvidia/Cosmos3-Super-Image2Video-4Step"
MINIMAX_H3_REPOSITORY = "MiniMaxAI/MiniMax-H3"

# One installed Modular pipeline class can have multiple official weight/config
# contracts. Keep the allowed repositories beside the pinned workflow truth;
# the loader still requires an immutable catalog revision for the selected
# repository, and the downstream action validates the workflow/repository pair.
PINNED_MODULAR_REPOSITORY_VARIANTS = {
    # Both official text-to-image checkpoints serialize the same
    # QwenImagePipeline component classes. They are reviewed model variants of
    # one Modular workflow, not separate task routes.
    "QwenImageModularPipeline": (
        QWEN_IMAGE_REPOSITORY,
        QWEN_IMAGE_2512_REPOSITORY,
    ),
    "Cosmos3DistilledModularPipeline": (
        COSMOS3_DISTILLED_T2I_REPOSITORY,
        COSMOS3_DISTILLED_I2V_REPOSITORY,
    ),
    "HunyuanVideo15ModularPipeline": (
        HUNYUAN_VIDEO_15_T2V_REPOSITORY,
        HUNYUAN_VIDEO_15_I2V_REPOSITORY,
    ),
    # One immutable Modular index contains both transformer partitions. The
    # loader's separately reviewed workflow ID selects ``transformer`` for
    # t2va/fl2va or ``transformer_ref`` for ref2va before components load.
    "MiniMaxH3ModularPipeline": (MINIMAX_H3_REPOSITORY,),
    "WanModularPipeline": (WAN_T2V_REPOSITORY, WAN_T2V_14B_REPOSITORY),
    "WanImage2VideoModularPipeline": (
        WAN_I2V_REPOSITORY,
        WAN_I2V_720P_REPOSITORY,
        WAN_FLF_REPOSITORY,
    ),
}

# Repository membership above is useful for immutable artifact review, but it
# is deliberately broader than a user-facing model switch: some Modular
# pipeline classes publish repositories for different workflows/modalities.
# Only entries in this workflow-scoped map may be selected without replacing
# the BlockDefinitionV2 graph. A missing entry means the registered artifact
# remains the sole choice for that workflow.
PINNED_MODULAR_WORKFLOW_REPOSITORY_VARIANTS = {
    ("QwenImageModularPipeline", "text2image"): (
        QWEN_IMAGE_REPOSITORY,
        QWEN_IMAGE_2512_REPOSITORY,
    ),
}
# Weight filename variants are a separate contract from the repository choices
# above.  ModularPipeline.load_components() accepts ``variant`` as a loading
# override for every component.  Keep the reviewed override repository-scoped
# so a float16 dtype never guesses that an unrelated checkpoint publishes
# ``*.fp16.safetensors`` files.
PINNED_MODULAR_REPOSITORY_WEIGHT_VARIANTS = {
    "StableDiffusionXLModularPipeline": {
        "stabilityai/stable-diffusion-xl-base-1.0": "fp16",
    },
}


def reviewed_modular_weight_variant(model_type: str, repository: str) -> str | None:
    """Return the exact reviewed component filename variant for one artifact."""

    variants = PINNED_MODULAR_REPOSITORY_WEIGHT_VARIANTS.get(str(model_type), {})
    return variants.get(str(repository))


# Standard Hub indexes name the concrete classes serialized by each checkpoint,
# while the installed Modular blocks declare their reviewed base/factory types.
# These are exact repository-scoped aliases, not general subclass admission.
PINNED_MODULAR_REPOSITORY_COMPONENT_TYPES = {
    # The official Cosmos 3 indexes serialize the concrete fast tokenizer,
    # while the pinned Modular blocks deliberately request AutoTokenizer.
    # Keep that factory-to-concrete compatibility exact and repository scoped.
    COSMOS3_DISTILLED_T2I_REPOSITORY: {
        "text_tokenizer": ("transformers", "Qwen2TokenizerFast"),
    },
    COSMOS3_DISTILLED_I2V_REPOSITORY: {
        "text_tokenizer": ("transformers", "Qwen2TokenizerFast"),
    },
    # Transformers 5 consolidates legacy ``*TokenizerFast`` exports into the
    # corresponding tokenizers-backed class. These pinned checkpoint indexes
    # still serialize the legacy names, while both official symbols resolve to
    # the same installed class. Keep each compatibility exact and repository-
    # scoped instead of weakening component validation globally.
    FLUX_DEV_REPOSITORY: {
        "tokenizer_2": ("transformers", "T5TokenizerFast"),
    },
    FLUX_KONTEXT_REPOSITORY: {
        "tokenizer_2": ("transformers", "T5TokenizerFast"),
    },
    FLUX2_KLEIN_REPOSITORY: {
        "tokenizer": ("transformers", "Qwen2TokenizerFast"),
    },
    FLUX2_KLEIN_BASE_REPOSITORY: {
        "tokenizer": ("transformers", "Qwen2TokenizerFast"),
    },
    HUNYUAN_VIDEO_15_T2V_REPOSITORY: {
        "tokenizer": ("transformers", "Qwen2TokenizerFast"),
    },
    HUNYUAN_VIDEO_15_I2V_REPOSITORY: {
        "tokenizer": ("transformers", "Qwen2TokenizerFast"),
    },
    # The exact conversion source at PINNED_DIFFUSERS_REVISION writes the fast
    # concrete tokenizer into modular_model_index.json, while the H3 blocks
    # declare the public Qwen2Tokenizer base. Keep that one alias repository
    # scoped; all other H3 components already match the index exactly.
    MINIMAX_H3_REPOSITORY: {
        "tokenizer": ("transformers", "Qwen2TokenizerFast"),
    },
    WAN_T2V_REPOSITORY: {
        "tokenizer": ("transformers", "T5TokenizerFast"),
    },
    WAN_T2V_14B_REPOSITORY: {
        "tokenizer": ("transformers", "T5TokenizerFast"),
    },
    WAN_I2V_REPOSITORY: {
        "tokenizer": ("transformers", "T5TokenizerFast"),
        "image_encoder": ("transformers", "CLIPVisionModelWithProjection"),
    },
    WAN_I2V_720P_REPOSITORY: {
        "tokenizer": ("transformers", "T5TokenizerFast"),
        "image_encoder": ("transformers", "CLIPVisionModelWithProjection"),
    },
    WAN_FLF_REPOSITORY: {
        "tokenizer": ("transformers", "T5TokenizerFast"),
        "image_processor": ("transformers", "CLIPProcessor"),
        "image_encoder": ("transformers", "CLIPVisionModelWithProjection"),
    },
    # The official Wan Animate 2 Modular blocks deliberately type these as the
    # AutoTokenizer factory and SchedulerMixin base, while the pinned Hub
    # indexes serialize the concrete UMT5 tokenizer and variant-specific
    # schedulers (DPM-Solver for base, flow-match for Distilled).
    # Admit only these exact repository/component pairs; arbitrary factory or
    # base-class substitutions remain rejected.
    WAN_ANIMATE_2_REPOSITORY: {
        "tokenizer": ("transformers", "T5TokenizerFast"),
        "scheduler": ("diffusers", "DPMSolverMultistepScheduler"),
    },
    WAN_ANIMATE_2_DISTILLED_REPOSITORY: {
        "tokenizer": ("transformers", "T5TokenizerFast"),
        "scheduler": ("diffusers", "FlowMatchEulerDiscreteScheduler"),
    },
}
# A standard Diffusers repository can contain components used by its monolithic
# pipeline class that are not consumed by the mapped ModularPipeline blocks.
# Upstream ModularPipeline deliberately ignores those unmatched keys when it
# converts model_index.json into component specs. Keep MoDiff's stricter cache
# validation, but admit only reviewed repository-scoped unused declarations.
PINNED_MODULAR_REPOSITORY_IGNORED_STANDARD_COMPONENT_TYPES = {
    "Qwen/Qwen-Image-Edit": {
        "tokenizer": ("transformers", "Qwen2Tokenizer"),
    },
    "Qwen/Qwen-Image-Edit-2511": {
        "tokenizer": ("transformers", "Qwen2Tokenizer"),
    },
}
# A reviewed Modular index may publish a repository-owned component that is
# not consumed by the selected blocks. The Distilled T2I checkpoint includes
# the Cosmos audio tokenizer even though its exact text2image workflow returns
# vision only. Admit only this exact index declaration; arbitrary extra
# executable components remain rejected by ModelsLoader.
PINNED_MODULAR_REPOSITORY_ADDITIONAL_MODULAR_COMPONENT_TYPES = {
    COSMOS3_DISTILLED_T2I_REPOSITORY: {
        "sound_tokenizer": ("diffusers", "Cosmos3AVAEAudioTokenizer"),
    },
}
# The FLF standard pipeline serializes a combined processor wrapper even though
# its image_processor subfolder contains only the reviewed CLIP image processor
# config consumed by the installed Modular image block. Validate the serialized
# declaration above, then normalize this one load contract to the block type.
PINNED_MODULAR_REPOSITORY_LOAD_COMPONENT_TYPES = {
    WAN_FLF_REPOSITORY: {
        "image_processor": ("transformers", "CLIPImageProcessor"),
    },
}
WAN_WORKFLOW_REPOSITORIES = (
    ("image2video", WAN_I2V_REPOSITORY),
    ("image2video", WAN_I2V_720P_REPOSITORY),
    ("flf2v", WAN_FLF_REPOSITORY),
)


@dataclass(frozen=True)
class UpstreamWorkflowTruth:
    """One exact entry from an upstream AutoBlocks ``_workflow_map``."""

    name: str
    required_inputs: frozenset[str]


@dataclass(frozen=True)
class StateEdgeTruth:
    """A value that must be wireable between two generic MoDiff actions."""

    producer_action: str
    producer_output: str
    consumer_action: str
    consumer_input: str


@dataclass(frozen=True)
class ModularModeTruth:
    """A public mode backed by a constructible current Modular node contract."""

    upstream_workflow: str | None
    required_upstream_inputs: frozenset[str]
    action_sequence: tuple[str, ...]
    state_edges: tuple[StateEdgeTruth, ...]
    upstream_block_sequence: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModularStateFlowTruth:
    """A reviewed internal action flow that does not advertise a public mode."""

    upstream_workflow: str
    required_upstream_inputs: frozenset[str]
    upstream_block_sequence: tuple[str, ...]
    action_sequence: tuple[str, ...]
    state_edges: tuple[StateEdgeTruth, ...]


@dataclass(frozen=True)
class PinnedModularPipelineTruth:
    """Pinned upstream surface plus the narrower executable MoDiff surface."""

    blocks_class: str
    workflows: tuple[UpstreamWorkflowTruth, ...] = ()
    fixed_block_sequence: tuple[str, ...] = ()
    constructor_config: tuple[tuple[str, object], ...] = ()
    modes: tuple[tuple[str, ModularModeTruth], ...] = ()
    state_flows: tuple[tuple[str, ModularStateFlowTruth], ...] = ()

    def mode(self, name: str) -> ModularModeTruth | None:
        return dict(self.modes).get(name)

    def state_flow(self, name: str) -> ModularStateFlowTruth | None:
        return dict(self.state_flows).get(name)


def _workflow(name: str, *required_inputs: str) -> UpstreamWorkflowTruth:
    return UpstreamWorkflowTruth(name, frozenset(required_inputs))


def _helios_workflow_truth(blocks_class: str) -> PinnedModularPipelineTruth:
    text_actions = ("workflow_text_encoder", "workflow_video_denoise", "workflow_video_decoder")
    image_actions = (
        "workflow_text_encoder",
        "workflow_video_image_encoder",
        "workflow_video_denoise",
        "workflow_video_decoder",
    )
    video_actions = (
        "workflow_text_encoder",
        "workflow_video_encoder",
        "workflow_video_denoise",
        "workflow_video_decoder",
    )

    def state_edges(actions: tuple[str, ...]) -> tuple[StateEdgeTruth, ...]:
        return tuple(
            StateEdgeTruth(producer, "state_out", consumer, "state_in")
            for producer, consumer in zip(actions, actions[1:])
        )

    return PinnedModularPipelineTruth(
        blocks_class=blocks_class,
        workflows=(
            _workflow("text2video", "prompt"),
            _workflow("image2video", "image", "prompt"),
            _workflow("video2video", "prompt", "video"),
        ),
        modes=(
            (
                "text_to_video",
                ModularModeTruth(
                    "text2video",
                    frozenset({"prompt"}),
                    text_actions,
                    state_edges(text_actions),
                    ("text_encoder", "denoise", "decode"),
                ),
            ),
            (
                "image_to_video",
                ModularModeTruth(
                    "image2video",
                    frozenset({"image", "prompt"}),
                    image_actions,
                    state_edges(image_actions),
                    ("text_encoder", "vae_encoder", "denoise", "decode"),
                ),
            ),
            (
                "video_to_video",
                ModularModeTruth(
                    "video2video",
                    frozenset({"prompt", "video"}),
                    video_actions,
                    state_edges(video_actions),
                    ("text_encoder", "vae_encoder", "denoise", "decode"),
                ),
            ),
        ),
    )


def _hunyuan_video_15_workflow_truth() -> PinnedModularPipelineTruth:
    text_actions = (
        "workflow_hunyuan_video15_text_encoder",
        "workflow_hunyuan_video15_denoise",
        "workflow_hunyuan_video15_decoder",
    )
    image_actions = (
        "workflow_hunyuan_video15_text_encoder",
        "workflow_hunyuan_video15_vae_encoder",
        "workflow_hunyuan_video15_image_encoder",
        "workflow_hunyuan_video15_denoise",
        "workflow_hunyuan_video15_decoder",
    )

    def state_edges(actions: tuple[str, ...]) -> tuple[StateEdgeTruth, ...]:
        return tuple(
            StateEdgeTruth(producer, "state_out", consumer, "state_in")
            for producer, consumer in zip(actions, actions[1:])
        )

    return PinnedModularPipelineTruth(
        blocks_class="HunyuanVideo15AutoBlocks",
        workflows=(
            _workflow("text2video", "prompt"),
            _workflow("image2video", "image", "prompt"),
        ),
        modes=(
            (
                "text_to_video",
                ModularModeTruth(
                    "text2video",
                    frozenset({"prompt"}),
                    text_actions,
                    state_edges(text_actions),
                    (
                        "text_encoder",
                        "denoise.input",
                        "denoise.set_timesteps",
                        "denoise.prepare_latents",
                        "denoise.denoise",
                        "denoise.denoise.before_denoiser",
                        "denoise.denoise.denoiser",
                        "denoise.denoise.after_denoiser",
                        "decode",
                    ),
                ),
            ),
            (
                "image_to_video",
                ModularModeTruth(
                    "image2video",
                    frozenset({"image", "prompt"}),
                    image_actions,
                    state_edges(image_actions),
                    (
                        "text_encoder",
                        "vae_encoder",
                        "image_encoder",
                        "denoise.input",
                        "denoise.set_timesteps",
                        "denoise.prepare_latents",
                        "denoise.prepare_i2v_latents",
                        "denoise.denoise",
                        "denoise.denoise.before_denoiser",
                        "denoise.denoise.denoiser",
                        "denoise.denoise.after_denoiser",
                        "decode",
                    ),
                ),
            ),
        ),
    )


def _wan_animate_2_workflow_truth(blocks_class: str) -> PinnedModularPipelineTruth:
    actions = (
        "workflow_wan_animate_text_encoder",
        "workflow_wan_animate_image_encoder",
        "workflow_wan_animate_video_encoder",
        "workflow_wan_animate_vae_encoder",
        "workflow_wan_animate_denoise",
        "workflow_wan_animate_decoder",
    )
    blocks = ("text_encoder", "image_encoder", "video_encoder", "vae_encoder", "denoise", "decode")
    return PinnedModularPipelineTruth(
        blocks_class=blocks_class,
        fixed_block_sequence=blocks,
        modes=(
            (
                "character_animate",
                ModularModeTruth(
                    None,
                    frozenset({"driving_video", "image", "prompt"}),
                    actions,
                    tuple(
                        StateEdgeTruth(producer, "state_out", consumer, "state_in")
                        for producer, consumer in zip(actions, actions[1:])
                    ),
                    blocks,
                ),
            ),
        ),
    )


_TEXT_TO_OUTPUT_EDGES = (
    StateEdgeTruth("text_encoder", "embeddings", "denoise", "embeddings"),
    StateEdgeTruth("denoise", "latents", "decoder", "latents"),
)

_IMAGE_TO_OUTPUT_EDGES = (
    StateEdgeTruth("text_encoder", "embeddings", "denoise", "embeddings"),
    StateEdgeTruth("vae_encoder", "image_latents", "denoise", "image_latents"),
    StateEdgeTruth("denoise", "latents", "decoder", "latents"),
)

# SDXL keeps its tensor-valued state on ordinary typed graph edges while the
# opaque route binds generator continuation, component provenance, and Decode.
_SDXL_ROUTE_TEXT_TO_OUTPUT_EDGES = (
    StateEdgeTruth("text_encoder", "embeddings", "denoise", "embeddings"),
    StateEdgeTruth("denoise", "latents", "decoder", "latents"),
    StateEdgeTruth("denoise", "route_state_out", "decoder", "route_state_in"),
)

_SDXL_ROUTE_IMAGE_TO_OUTPUT_EDGES = (
    StateEdgeTruth("text_encoder", "embeddings", "denoise", "embeddings"),
    StateEdgeTruth("vae_encoder", "image_latents", "denoise", "image_latents"),
    StateEdgeTruth("vae_encoder", "route_state_out", "denoise", "route_state_in"),
    StateEdgeTruth("denoise", "latents", "decoder", "latents"),
    StateEdgeTruth("denoise", "route_state_out", "decoder", "route_state_in"),
)

_SDXL_ROUTE_CONTROL_TO_OUTPUT_EDGES = (
    StateEdgeTruth("text_encoder", "embeddings", "denoise", "embeddings"),
    StateEdgeTruth("controlnet", "controlnet_bundle", "denoise", "controlnet_bundle"),
    StateEdgeTruth("denoise", "latents", "decoder", "latents"),
    StateEdgeTruth("denoise", "route_state_out", "decoder", "route_state_in"),
)

_SDXL_ROUTE_INPAINT_TO_OUTPUT_EDGES = (
    StateEdgeTruth("text_encoder", "embeddings", "denoise", "embeddings"),
    StateEdgeTruth("vae_encoder", "image_latents", "denoise", "image_latents"),
    StateEdgeTruth("vae_encoder", "mask", "denoise", "mask"),
    StateEdgeTruth(
        "vae_encoder",
        "masked_image_latents",
        "denoise",
        "masked_image_latents",
    ),
    StateEdgeTruth("vae_encoder", "route_state_out", "denoise", "route_state_in"),
    StateEdgeTruth("denoise", "latents", "decoder", "latents"),
    StateEdgeTruth("denoise", "route_state_out", "decoder", "route_state_in"),
)

_SDXL_ROUTE_CONTROL_IMAGE_TO_OUTPUT_EDGES = (
    StateEdgeTruth("text_encoder", "embeddings", "denoise", "embeddings"),
    StateEdgeTruth("vae_encoder", "image_latents", "denoise", "image_latents"),
    StateEdgeTruth("vae_encoder", "route_state_out", "denoise", "route_state_in"),
    StateEdgeTruth("controlnet", "controlnet_bundle", "denoise", "controlnet_bundle"),
    StateEdgeTruth("denoise", "latents", "decoder", "latents"),
    StateEdgeTruth("denoise", "route_state_out", "decoder", "route_state_in"),
)

_SDXL_ROUTE_CONTROL_INPAINT_TO_OUTPUT_EDGES = (
    StateEdgeTruth("text_encoder", "embeddings", "denoise", "embeddings"),
    StateEdgeTruth("vae_encoder", "image_latents", "denoise", "image_latents"),
    StateEdgeTruth("vae_encoder", "mask", "denoise", "mask"),
    StateEdgeTruth(
        "vae_encoder",
        "masked_image_latents",
        "denoise",
        "masked_image_latents",
    ),
    StateEdgeTruth("vae_encoder", "route_state_out", "denoise", "route_state_in"),
    StateEdgeTruth("controlnet", "controlnet_bundle", "denoise", "controlnet_bundle"),
    StateEdgeTruth("denoise", "latents", "decoder", "latents"),
    StateEdgeTruth("denoise", "route_state_out", "decoder", "route_state_in"),
)

_SDXL_IP_ADAPTER_EDGE = (StateEdgeTruth("ip_adapter", "ip_adapter", "denoise", "ip_adapter"),)

_SDXL_INPAINT_BLOCK_SEQUENCE = (
    "text_encoder",
    "vae_encoder",
    "denoise.input",
    "denoise.before_denoise.set_timesteps",
    "denoise.before_denoise.prepare_latents",
    "denoise.before_denoise.prepare_add_cond",
    "denoise.denoise",
    "decode",
)

_SDXL_CONTROLNET_BLOCK_SEQUENCE = (
    "text_encoder",
    "vae_encoder",
    "denoise.input",
    "denoise.before_denoise.set_timesteps",
    "denoise.before_denoise.prepare_latents",
    "denoise.before_denoise.prepare_add_cond",
    "denoise.controlnet_input",
    "denoise.denoise",
    "decode",
)

_SDXL_IP_ADAPTER_BLOCK_SEQUENCE = (
    "text_encoder",
    "ip_adapter",
    "vae_encoder",
    "denoise.input",
    "denoise.before_denoise.set_timesteps",
    "denoise.before_denoise.prepare_latents",
    "denoise.before_denoise.prepare_add_cond",
    "denoise.denoise",
    "decode",
)

_SDXL_IP_ADAPTER_TEXT_BLOCK_SEQUENCE = (
    "text_encoder",
    "ip_adapter",
    "denoise.input",
    "denoise.before_denoise.set_timesteps",
    "denoise.before_denoise.prepare_latents",
    "denoise.before_denoise.prepare_add_cond",
    "denoise.denoise",
    "decode",
)

_SDXL_IP_ADAPTER_CONTROL_BLOCK_SEQUENCE = (
    "text_encoder",
    "ip_adapter",
    "vae_encoder",
    "denoise.input",
    "denoise.before_denoise.set_timesteps",
    "denoise.before_denoise.prepare_latents",
    "denoise.before_denoise.prepare_add_cond",
    "denoise.controlnet_input",
    "denoise.denoise",
    "decode",
)

_SDXL_IP_ADAPTER_CONTROL_TEXT_BLOCK_SEQUENCE = (
    "text_encoder",
    "ip_adapter",
    "denoise.input",
    "denoise.before_denoise.set_timesteps",
    "denoise.before_denoise.prepare_latents",
    "denoise.before_denoise.prepare_add_cond",
    "denoise.controlnet_input",
    "denoise.denoise",
    "decode",
)

_QWEN_ROUTE_IMAGE_TO_OUTPUT_EDGES = (
    StateEdgeTruth("text_encoder", "embeddings", "denoise", "embeddings"),
    StateEdgeTruth("vae_encoder", "image_latents", "denoise", "image_latents"),
    StateEdgeTruth("vae_encoder", "route_state_out", "denoise", "route_state_in"),
    StateEdgeTruth("denoise", "latents", "decoder", "latents"),
    StateEdgeTruth("denoise", "route_state_out", "decoder", "route_state_in"),
)

_QWEN_ROUTE_TEXT_TO_OUTPUT_EDGES = (
    StateEdgeTruth("text_encoder", "embeddings", "denoise", "embeddings"),
    StateEdgeTruth("denoise", "latents", "decoder", "latents"),
    StateEdgeTruth("denoise", "route_state_out", "decoder", "route_state_in"),
)

_QWEN_TEXT2IMAGE_BLOCK_SEQUENCE = (
    "text_encoder",
    "denoise.input",
    "denoise.prepare_latents",
    "denoise.set_timesteps",
    "denoise.prepare_rope_inputs",
    "denoise.denoise",
    "denoise.denoise.before_denoiser",
    "denoise.denoise.denoiser",
    "denoise.denoise.after_denoiser",
    "denoise.after_denoise",
    "decode.decode",
    "decode.postprocess",
)

_CONTROL_TO_OUTPUT_EDGES = (
    StateEdgeTruth("text_encoder", "embeddings", "denoise", "embeddings"),
    StateEdgeTruth("controlnet", "controlnet_bundle", "denoise", "controlnet_bundle"),
    StateEdgeTruth("denoise", "latents", "decoder", "latents"),
)

# Qwen ControlNet seals its post-control-VAE generator for Denoise, and Decode
# remains loader-bound even when no image VAE route precedes ControlNet.
_QWEN_CONTROL_TO_OUTPUT_EDGES = (
    StateEdgeTruth("text_encoder", "embeddings", "denoise", "embeddings"),
    StateEdgeTruth("controlnet", "controlnet_bundle", "denoise", "controlnet_bundle"),
    StateEdgeTruth("controlnet", "route_state_out", "denoise", "route_state_in"),
    StateEdgeTruth("denoise", "latents", "decoder", "latents"),
    StateEdgeTruth("denoise", "route_state_out", "decoder", "route_state_in"),
)

# Main image latents remain a direct typed edge into Denoise. The opaque main
# VAE route instead passes through ControlNet so its post-control-VAE generator,
# mask/overlay state, and exact control-latent identity reach Denoise together.
_QWEN_ROUTE_CONTROL_IMAGE_TO_OUTPUT_EDGES = (
    StateEdgeTruth("text_encoder", "embeddings", "denoise", "embeddings"),
    StateEdgeTruth("vae_encoder", "image_latents", "denoise", "image_latents"),
    StateEdgeTruth("vae_encoder", "route_state_out", "controlnet", "route_state_in"),
    StateEdgeTruth("controlnet", "controlnet_bundle", "denoise", "controlnet_bundle"),
    StateEdgeTruth("controlnet", "route_state_out", "denoise", "route_state_in"),
    StateEdgeTruth("denoise", "latents", "decoder", "latents"),
    StateEdgeTruth("denoise", "route_state_out", "decoder", "route_state_in"),
)

_QWEN_IMAGE2IMAGE_BLOCK_SEQUENCE = (
    "text_encoder",
    "vae_encoder.preprocess",
    "vae_encoder.encode",
    "denoise.input.text_inputs",
    "denoise.input.additional_inputs",
    "denoise.prepare_latents",
    "denoise.set_timesteps",
    "denoise.prepare_img2img_latents",
    "denoise.prepare_rope_inputs",
    "denoise.denoise",
    "denoise.after_denoise",
    "decode.decode",
    "decode.postprocess",
)

_QWEN_INPAINT_BLOCK_SEQUENCE = (
    "text_encoder",
    "vae_encoder.preprocess",
    "vae_encoder.encode",
    "denoise.input.text_inputs",
    "denoise.input.additional_inputs",
    "denoise.prepare_latents",
    "denoise.set_timesteps",
    "denoise.prepare_inpaint_latents.add_noise_to_latents",
    "denoise.prepare_inpaint_latents.create_mask_latents",
    "denoise.prepare_rope_inputs",
    "denoise.denoise",
    "denoise.after_denoise",
    "decode.decode",
    "decode.postprocess",
)

_QWEN_CONTROL_IMAGE2IMAGE_BLOCK_SEQUENCE = (
    "text_encoder",
    "vae_encoder.preprocess",
    "vae_encoder.encode",
    "controlnet_vae_encoder",
    "denoise.input.text_inputs",
    "denoise.input.additional_inputs",
    "denoise.controlnet_input",
    "denoise.prepare_latents",
    "denoise.set_timesteps",
    "denoise.prepare_img2img_latents",
    "denoise.prepare_rope_inputs",
    "denoise.controlnet_before_denoise",
    "denoise.controlnet_denoise",
    "denoise.after_denoise",
    "decode.decode",
    "decode.postprocess",
)

_QWEN_CONTROL_INPAINT_BLOCK_SEQUENCE = (
    "text_encoder",
    "vae_encoder.preprocess",
    "vae_encoder.encode",
    "controlnet_vae_encoder",
    "denoise.input.text_inputs",
    "denoise.input.additional_inputs",
    "denoise.controlnet_input",
    "denoise.prepare_latents",
    "denoise.set_timesteps",
    "denoise.prepare_inpaint_latents.add_noise_to_latents",
    "denoise.prepare_inpaint_latents.create_mask_latents",
    "denoise.prepare_rope_inputs",
    "denoise.controlnet_before_denoise",
    "denoise.controlnet_denoise",
    "denoise.after_denoise",
    "decode.decode",
    "decode.postprocess",
)

_QWEN_EDIT_INPAINT_BLOCK_SEQUENCE = (
    "text_encoder.resize",
    "text_encoder.encode",
    "vae_encoder.resize",
    "vae_encoder.preprocess",
    "vae_encoder.encode",
    "denoise.input.text_inputs",
    "denoise.input.additional_inputs",
    "denoise.prepare_latents",
    "denoise.set_timesteps",
    "denoise.prepare_inpaint_latents.add_noise_to_latents",
    "denoise.prepare_inpaint_latents.create_mask_latents",
    "denoise.prepare_rope_inputs",
    "denoise.denoise",
    "denoise.denoise.before_denoiser",
    "denoise.denoise.denoiser",
    "denoise.denoise.after_denoiser",
    "denoise.denoise.after_denoiser_inpaint",
    "denoise.after_denoise",
    "decode.decode",
    "decode.postprocess",
)

_WAN_IMAGE_TO_OUTPUT_EDGES = (
    StateEdgeTruth("text_encoder", "embeddings", "denoise", "embeddings"),
    StateEdgeTruth("image_encoder", "image_embeds", "denoise", "image_embeds"),
    StateEdgeTruth("image_encoder", "route_state_out", "vae_encoder", "route_state_in"),
    StateEdgeTruth(
        "vae_encoder",
        "image_condition_latents",
        "denoise",
        "image_condition_latents",
    ),
    StateEdgeTruth("vae_encoder", "route_state_out", "denoise", "route_state_in"),
    StateEdgeTruth("denoise", "latents", "decoder", "latents"),
    StateEdgeTruth("denoise", "route_state_out", "decoder", "route_state_in"),
)

# The raw first/last-frame VAE latents are intentionally not an edge into
# Denoise.  Pinned Wan consumes only the prepared image_condition_latents.
_WAN_IMAGE2VIDEO_BLOCK_SEQUENCE = (
    "text_encoder",
    "image_encoder.image_resize",
    "image_encoder.image_encoder",
    "vae_encoder.image_resize",
    "vae_encoder.vae_encoder",
    "vae_encoder.prepare_first_frame_latents",
    "denoise.input",
    "denoise.additional_inputs",
    "denoise.set_timesteps",
    "denoise.prepare_latents",
    "denoise.denoise",
    "decode",
)

_WAN_FLF2V_BLOCK_SEQUENCE = (
    "text_encoder",
    "image_encoder.image_resize",
    "image_encoder.last_image_resize",
    "image_encoder.image_encoder",
    "vae_encoder.image_resize",
    "vae_encoder.last_image_resize",
    "vae_encoder.vae_encoder",
    "vae_encoder.prepare_first_last_frame_latents",
    "denoise.input",
    "denoise.additional_inputs",
    "denoise.set_timesteps",
    "denoise.prepare_latents",
    "denoise.denoise",
    "decode",
    )


def _cosmos3_omni_workflow_truth() -> PinnedModularPipelineTruth:
    actions = (
        "workflow_cosmos3_omni_text_encoder",
        "workflow_cosmos3_omni_denoise",
        "workflow_cosmos3_omni_decoder",
        "workflow_cosmos3_omni_after_decode",
    )
    edges = tuple(
        StateEdgeTruth(producer, "state_out", consumer, "state_in")
        for producer, consumer in zip(actions, actions[1:])
    )
    block_sequence = (
        "text_encoder",
        "denoise.prepare_text_segments",
        "denoise.prepare_vision_latents",
        "denoise.pack_vision_sequence",
        "denoise.prepare_vision_denoiser_inputs",
        "denoise.set_timesteps",
        "denoise.denoise",
        "denoise.denoise.prepare_vision",
        "denoise.denoise.denoiser",
        "denoise.denoise.update_vision",
        "decode.video",
        "after_decode",
    )
    return PinnedModularPipelineTruth(
        blocks_class="Cosmos3OmniBlocks",
        # Keep the complete pinned upstream dispatch map even though only the
        # two plain-text routes below have exact MoDiff execution contracts.
        workflows=(
            _workflow("text2image", "num_frames", "prompt"),
            _workflow("text2video", "prompt"),
            _workflow("image2video", "image", "prompt"),
            _workflow("video2video", "prompt", "video"),
            _workflow("text2video_with_sound", "enable_sound", "prompt"),
            _workflow("image2video_with_sound", "enable_sound", "image", "prompt"),
            _workflow("video2video_with_sound", "enable_sound", "prompt", "video"),
            _workflow("action_policy", "action", "prompt"),
            _workflow("action_forward_dynamics", "action", "prompt"),
            _workflow("action_inverse_dynamics", "action", "prompt"),
        ),
        modes=tuple(
            (
                mode,
                ModularModeTruth(
                    workflow,
                    frozenset({"num_inference_steps", "prompt"}),
                    actions,
                    edges,
                    block_sequence,
                ),
            )
            for mode, workflow in (
                ("text_to_image", "text2image"),
                ("text_to_video", "text2video"),
            )
        ),
    )


PINNED_MODULAR_WORKFLOW_TRUTH: dict[str, PinnedModularPipelineTruth] = {
    "AnimaModularPipeline": PinnedModularPipelineTruth(
        blocks_class="AnimaAutoBlocks",
        workflows=(
            _workflow("text2image", "prompt"),
            _workflow("img2img", "image", "prompt"),
        ),
        modes=(
            (
                "text_to_image",
                ModularModeTruth(
                    "text2image",
                    frozenset({"prompt"}),
                    ("workflow_text_encoder", "workflow_image_denoise", "workflow_image_decoder"),
                    (
                        StateEdgeTruth(
                            "workflow_text_encoder", "state_out", "workflow_image_denoise", "state_in"
                        ),
                        StateEdgeTruth(
                            "workflow_image_denoise", "state_out", "workflow_image_decoder", "state_in"
                        ),
                    ),
                    (
                        "text_encoder",
                        "denoise.text_conditioning",
                        "denoise.input",
                        "denoise.prepare_latents",
                        "denoise.set_timesteps",
                        "denoise.denoise",
                        "decode.decode",
                        "decode.postprocess",
                    ),
                ),
            ),
            (
                "image_to_image",
                ModularModeTruth(
                    "img2img",
                    frozenset({"image", "prompt"}),
                    (
                        "workflow_text_encoder",
                        "workflow_image_encoder",
                        "workflow_image_denoise",
                        "workflow_image_decoder",
                    ),
                    (
                        StateEdgeTruth(
                            "workflow_text_encoder", "state_out", "workflow_image_encoder", "state_in"
                        ),
                        StateEdgeTruth(
                            "workflow_image_encoder", "state_out", "workflow_image_denoise", "state_in"
                        ),
                        StateEdgeTruth(
                            "workflow_image_denoise", "state_out", "workflow_image_decoder", "state_in"
                        ),
                    ),
                    (
                        "text_encoder",
                        "vae_encoder",
                        "denoise.text_conditioning",
                        "denoise.input",
                        "denoise.image_input",
                        "denoise.set_timesteps",
                        "denoise.prepare_latents",
                        "denoise.denoise",
                        "decode.decode",
                        "decode.postprocess",
                    ),
                ),
            ),
        ),
    ),
    "HeliosModularPipeline": _helios_workflow_truth("HeliosAutoBlocks"),
    "HeliosPyramidModularPipeline": _helios_workflow_truth("HeliosPyramidAutoBlocks"),
    "HeliosPyramidDistilledModularPipeline": _helios_workflow_truth("HeliosPyramidDistilledAutoBlocks"),
    "HunyuanVideo15ModularPipeline": _hunyuan_video_15_workflow_truth(),
    "Cosmos3OmniModularPipeline": _cosmos3_omni_workflow_truth(),
    "WanAnimate2ModularPipeline": _wan_animate_2_workflow_truth("WanAnimate2Blocks"),
    "WanAnimate2DistilledModularPipeline": _wan_animate_2_workflow_truth("WanAnimate2DistilledBlocks"),
    "StableDiffusionXLModularPipeline": PinnedModularPipelineTruth(
        blocks_class="StableDiffusionXLAutoBlocks",
        workflows=(
            _workflow("text2image", "prompt"),
            _workflow("image2image", "image", "prompt"),
            _workflow("inpainting", "mask_image", "image", "prompt"),
            _workflow("controlnet_text2image", "control_image", "prompt"),
            _workflow("controlnet_image2image", "control_image", "image", "prompt"),
            _workflow("controlnet_inpainting", "control_image", "mask_image", "image", "prompt"),
            _workflow("controlnet_union_text2image", "control_image", "control_mode", "prompt"),
            _workflow(
                "controlnet_union_image2image",
                "control_image",
                "control_mode",
                "image",
                "prompt",
            ),
            _workflow(
                "controlnet_union_inpainting",
                "control_image",
                "control_mode",
                "mask_image",
                "image",
                "prompt",
            ),
            _workflow("ip_adapter_text2image", "ip_adapter_image", "prompt"),
            _workflow("ip_adapter_image2image", "ip_adapter_image", "image", "prompt"),
            _workflow("ip_adapter_inpainting", "ip_adapter_image", "mask_image", "image", "prompt"),
            _workflow(
                "ip_adapter_controlnet_text2image",
                "ip_adapter_image",
                "control_image",
                "prompt",
            ),
            _workflow(
                "ip_adapter_controlnet_image2image",
                "ip_adapter_image",
                "control_image",
                "image",
                "prompt",
            ),
            _workflow(
                "ip_adapter_controlnet_inpainting",
                "ip_adapter_image",
                "control_image",
                "mask_image",
                "image",
                "prompt",
            ),
            _workflow(
                "ip_adapter_controlnet_union_text2image",
                "ip_adapter_image",
                "control_image",
                "control_mode",
                "prompt",
            ),
            _workflow(
                "ip_adapter_controlnet_union_image2image",
                "ip_adapter_image",
                "control_image",
                "control_mode",
                "image",
                "prompt",
            ),
            _workflow(
                "ip_adapter_controlnet_union_inpainting",
                "ip_adapter_image",
                "control_image",
                "control_mode",
                "mask_image",
                "image",
                "prompt",
            ),
        ),
        modes=(
            (
                "text_to_image",
                ModularModeTruth(
                    "text2image",
                    frozenset({"prompt"}),
                    ("text_encoder", "denoise", "decoder"),
                    _SDXL_ROUTE_TEXT_TO_OUTPUT_EDGES,
                ),
            ),
            (
                "image_to_image",
                ModularModeTruth(
                    "image2image",
                    frozenset({"image", "prompt"}),
                    ("text_encoder", "vae_encoder", "denoise", "decoder"),
                    _SDXL_ROUTE_IMAGE_TO_OUTPUT_EDGES,
                ),
            ),
            (
                "control_image",
                ModularModeTruth(
                    "controlnet_text2image",
                    frozenset({"control_image", "prompt"}),
                    ("text_encoder", "controlnet", "denoise", "decoder"),
                    _SDXL_ROUTE_CONTROL_TO_OUTPUT_EDGES,
                ),
            ),
            (
                "control_edit_image",
                ModularModeTruth(
                    "controlnet_image2image",
                    frozenset({"control_image", "image", "prompt"}),
                    ("text_encoder", "vae_encoder", "controlnet", "denoise", "decoder"),
                    _SDXL_ROUTE_CONTROL_IMAGE_TO_OUTPUT_EDGES,
                    _SDXL_CONTROLNET_BLOCK_SEQUENCE,
                ),
            ),
            (
                "control_inpaint",
                ModularModeTruth(
                    "controlnet_inpainting",
                    frozenset({"control_image", "mask_image", "image", "prompt"}),
                    ("text_encoder", "vae_encoder", "controlnet", "denoise", "decoder"),
                    _SDXL_ROUTE_CONTROL_INPAINT_TO_OUTPUT_EDGES,
                    _SDXL_CONTROLNET_BLOCK_SEQUENCE,
                ),
            ),
            (
                "inpaint",
                ModularModeTruth(
                    "inpainting",
                    frozenset({"mask_image", "image", "prompt"}),
                    ("text_encoder", "vae_encoder", "denoise", "decoder"),
                    _SDXL_ROUTE_INPAINT_TO_OUTPUT_EDGES,
                    _SDXL_INPAINT_BLOCK_SEQUENCE,
                ),
            ),
        ),
        state_flows=(
            (
                "inpainting",
                ModularStateFlowTruth(
                    "inpainting",
                    frozenset({"mask_image", "image", "prompt"}),
                    _SDXL_INPAINT_BLOCK_SEQUENCE,
                    ("text_encoder", "vae_encoder", "denoise", "decoder"),
                    _SDXL_ROUTE_INPAINT_TO_OUTPUT_EDGES,
                ),
            ),
            (
                "controlnet_image2image",
                ModularStateFlowTruth(
                    "controlnet_image2image",
                    frozenset({"control_image", "image", "prompt"}),
                    _SDXL_CONTROLNET_BLOCK_SEQUENCE,
                    ("text_encoder", "vae_encoder", "controlnet", "denoise", "decoder"),
                    _SDXL_ROUTE_CONTROL_IMAGE_TO_OUTPUT_EDGES,
                ),
            ),
            (
                "controlnet_inpainting",
                ModularStateFlowTruth(
                    "controlnet_inpainting",
                    frozenset({"control_image", "mask_image", "image", "prompt"}),
                    _SDXL_CONTROLNET_BLOCK_SEQUENCE,
                    ("text_encoder", "vae_encoder", "controlnet", "denoise", "decoder"),
                    _SDXL_ROUTE_CONTROL_INPAINT_TO_OUTPUT_EDGES,
                ),
            ),
            (
                "controlnet_union_text2image",
                ModularStateFlowTruth(
                    "controlnet_union_text2image",
                    frozenset({"control_image", "control_mode", "prompt"}),
                    _SDXL_CONTROLNET_BLOCK_SEQUENCE,
                    ("text_encoder", "controlnet", "denoise", "decoder"),
                    _SDXL_ROUTE_CONTROL_TO_OUTPUT_EDGES,
                ),
            ),
            (
                "controlnet_union_image2image",
                ModularStateFlowTruth(
                    "controlnet_union_image2image",
                    frozenset({"control_image", "control_mode", "image", "prompt"}),
                    _SDXL_CONTROLNET_BLOCK_SEQUENCE,
                    ("text_encoder", "vae_encoder", "controlnet", "denoise", "decoder"),
                    _SDXL_ROUTE_CONTROL_IMAGE_TO_OUTPUT_EDGES,
                ),
            ),
            (
                "controlnet_union_inpainting",
                ModularStateFlowTruth(
                    "controlnet_union_inpainting",
                    frozenset({"control_image", "control_mode", "mask_image", "image", "prompt"}),
                    _SDXL_CONTROLNET_BLOCK_SEQUENCE,
                    ("text_encoder", "vae_encoder", "controlnet", "denoise", "decoder"),
                    _SDXL_ROUTE_CONTROL_INPAINT_TO_OUTPUT_EDGES,
                ),
            ),
            (
                "ip_adapter_text2image",
                ModularStateFlowTruth(
                    "ip_adapter_text2image",
                    frozenset({"ip_adapter_image", "prompt"}),
                    _SDXL_IP_ADAPTER_TEXT_BLOCK_SEQUENCE,
                    ("text_encoder", "ip_adapter", "denoise", "decoder"),
                    _SDXL_ROUTE_TEXT_TO_OUTPUT_EDGES + _SDXL_IP_ADAPTER_EDGE,
                ),
            ),
            (
                "ip_adapter_image2image",
                ModularStateFlowTruth(
                    "ip_adapter_image2image",
                    frozenset({"ip_adapter_image", "image", "prompt"}),
                    _SDXL_IP_ADAPTER_BLOCK_SEQUENCE,
                    ("text_encoder", "ip_adapter", "vae_encoder", "denoise", "decoder"),
                    _SDXL_ROUTE_IMAGE_TO_OUTPUT_EDGES + _SDXL_IP_ADAPTER_EDGE,
                ),
            ),
            (
                "ip_adapter_inpainting",
                ModularStateFlowTruth(
                    "ip_adapter_inpainting",
                    frozenset({"ip_adapter_image", "mask_image", "image", "prompt"}),
                    _SDXL_IP_ADAPTER_BLOCK_SEQUENCE,
                    ("text_encoder", "ip_adapter", "vae_encoder", "denoise", "decoder"),
                    _SDXL_ROUTE_INPAINT_TO_OUTPUT_EDGES + _SDXL_IP_ADAPTER_EDGE,
                ),
            ),
            (
                "ip_adapter_controlnet_text2image",
                ModularStateFlowTruth(
                    "ip_adapter_controlnet_text2image",
                    frozenset({"ip_adapter_image", "control_image", "prompt"}),
                    _SDXL_IP_ADAPTER_CONTROL_TEXT_BLOCK_SEQUENCE,
                    ("text_encoder", "ip_adapter", "controlnet", "denoise", "decoder"),
                    _SDXL_ROUTE_CONTROL_TO_OUTPUT_EDGES + _SDXL_IP_ADAPTER_EDGE,
                ),
            ),
            (
                "ip_adapter_controlnet_image2image",
                ModularStateFlowTruth(
                    "ip_adapter_controlnet_image2image",
                    frozenset({"ip_adapter_image", "control_image", "image", "prompt"}),
                    _SDXL_IP_ADAPTER_CONTROL_BLOCK_SEQUENCE,
                    ("text_encoder", "ip_adapter", "vae_encoder", "controlnet", "denoise", "decoder"),
                    _SDXL_ROUTE_CONTROL_IMAGE_TO_OUTPUT_EDGES + _SDXL_IP_ADAPTER_EDGE,
                ),
            ),
            (
                "ip_adapter_controlnet_inpainting",
                ModularStateFlowTruth(
                    "ip_adapter_controlnet_inpainting",
                    frozenset({"ip_adapter_image", "control_image", "mask_image", "image", "prompt"}),
                    _SDXL_IP_ADAPTER_CONTROL_BLOCK_SEQUENCE,
                    ("text_encoder", "ip_adapter", "vae_encoder", "controlnet", "denoise", "decoder"),
                    _SDXL_ROUTE_CONTROL_INPAINT_TO_OUTPUT_EDGES + _SDXL_IP_ADAPTER_EDGE,
                ),
            ),
            (
                "ip_adapter_controlnet_union_text2image",
                ModularStateFlowTruth(
                    "ip_adapter_controlnet_union_text2image",
                    frozenset({"ip_adapter_image", "control_image", "control_mode", "prompt"}),
                    _SDXL_IP_ADAPTER_CONTROL_TEXT_BLOCK_SEQUENCE,
                    ("text_encoder", "ip_adapter", "controlnet", "denoise", "decoder"),
                    _SDXL_ROUTE_CONTROL_TO_OUTPUT_EDGES + _SDXL_IP_ADAPTER_EDGE,
                ),
            ),
            (
                "ip_adapter_controlnet_union_image2image",
                ModularStateFlowTruth(
                    "ip_adapter_controlnet_union_image2image",
                    frozenset({"ip_adapter_image", "control_image", "control_mode", "image", "prompt"}),
                    _SDXL_IP_ADAPTER_CONTROL_BLOCK_SEQUENCE,
                    ("text_encoder", "ip_adapter", "vae_encoder", "controlnet", "denoise", "decoder"),
                    _SDXL_ROUTE_CONTROL_IMAGE_TO_OUTPUT_EDGES + _SDXL_IP_ADAPTER_EDGE,
                ),
            ),
            (
                "ip_adapter_controlnet_union_inpainting",
                ModularStateFlowTruth(
                    "ip_adapter_controlnet_union_inpainting",
                    frozenset({"ip_adapter_image", "control_image", "control_mode", "mask_image", "image", "prompt"}),
                    _SDXL_IP_ADAPTER_CONTROL_BLOCK_SEQUENCE,
                    ("text_encoder", "ip_adapter", "vae_encoder", "controlnet", "denoise", "decoder"),
                    _SDXL_ROUTE_CONTROL_INPAINT_TO_OUTPUT_EDGES + _SDXL_IP_ADAPTER_EDGE,
                ),
            ),
        ),
    ),
    "QwenImageModularPipeline": PinnedModularPipelineTruth(
        blocks_class="QwenImageAutoBlocks",
        workflows=(
            _workflow("text2image", "prompt"),
            _workflow("image2image", "prompt", "image"),
            _workflow("inpainting", "prompt", "mask_image", "image"),
            _workflow("controlnet_text2image", "prompt", "control_image"),
            _workflow("controlnet_image2image", "prompt", "image", "control_image"),
            _workflow("controlnet_inpainting", "prompt", "mask_image", "image", "control_image"),
        ),
        modes=(
            (
                "text_to_image",
                ModularModeTruth(
                    "text2image",
                    frozenset({"prompt"}),
                    ("text_encoder", "denoise", "decoder"),
                    _QWEN_ROUTE_TEXT_TO_OUTPUT_EDGES,
                    _QWEN_TEXT2IMAGE_BLOCK_SEQUENCE,
                ),
            ),
            (
                "control_image",
                ModularModeTruth(
                    "controlnet_text2image",
                    frozenset({"prompt", "control_image"}),
                    ("text_encoder", "controlnet", "denoise", "decoder"),
                    _QWEN_CONTROL_TO_OUTPUT_EDGES,
                ),
            ),
        ),
        state_flows=(
            (
                "image2image",
                ModularStateFlowTruth(
                    "image2image",
                    frozenset({"prompt", "image"}),
                    _QWEN_IMAGE2IMAGE_BLOCK_SEQUENCE,
                    ("text_encoder", "vae_encoder", "denoise", "decoder"),
                    _QWEN_ROUTE_IMAGE_TO_OUTPUT_EDGES,
                ),
            ),
            (
                "inpainting",
                ModularStateFlowTruth(
                    "inpainting",
                    frozenset({"prompt", "mask_image", "image"}),
                    _QWEN_INPAINT_BLOCK_SEQUENCE,
                    ("text_encoder", "vae_encoder", "denoise", "decoder"),
                    _QWEN_ROUTE_IMAGE_TO_OUTPUT_EDGES,
                ),
            ),
            (
                "controlnet_image2image",
                ModularStateFlowTruth(
                    "controlnet_image2image",
                    frozenset({"prompt", "image", "control_image"}),
                    _QWEN_CONTROL_IMAGE2IMAGE_BLOCK_SEQUENCE,
                    ("text_encoder", "vae_encoder", "controlnet", "denoise", "decoder"),
                    _QWEN_ROUTE_CONTROL_IMAGE_TO_OUTPUT_EDGES,
                ),
            ),
            (
                "controlnet_inpainting",
                ModularStateFlowTruth(
                    "controlnet_inpainting",
                    frozenset({"prompt", "mask_image", "image", "control_image"}),
                    _QWEN_CONTROL_INPAINT_BLOCK_SEQUENCE,
                    ("text_encoder", "vae_encoder", "controlnet", "denoise", "decoder"),
                    _QWEN_ROUTE_CONTROL_IMAGE_TO_OUTPUT_EDGES,
                ),
            ),
        ),
    ),
    "QwenImageEditModularPipeline": PinnedModularPipelineTruth(
        blocks_class="QwenImageEditAutoBlocks",
        workflows=(
            _workflow("image_conditioned", "prompt", "image"),
            _workflow("image_conditioned_inpainting", "prompt", "mask_image", "image"),
        ),
        modes=(
            (
                "edit_image",
                ModularModeTruth(
                    "image_conditioned",
                    frozenset({"prompt", "image"}),
                    ("text_encoder", "vae_encoder", "denoise", "decoder"),
                    _QWEN_ROUTE_IMAGE_TO_OUTPUT_EDGES,
                ),
            ),
        ),
        state_flows=(
            (
                "image_conditioned_inpainting",
                ModularStateFlowTruth(
                    "image_conditioned_inpainting",
                    frozenset({"prompt", "mask_image", "image"}),
                    _QWEN_EDIT_INPAINT_BLOCK_SEQUENCE,
                    ("text_encoder", "vae_encoder", "denoise", "decoder"),
                    _QWEN_ROUTE_IMAGE_TO_OUTPUT_EDGES,
                ),
            ),
        ),
    ),
    "QwenImageEditPlusModularPipeline": PinnedModularPipelineTruth(
        blocks_class="QwenImageEditPlusAutoBlocks",
        fixed_block_sequence=("text_encoder", "vae_encoder", "denoise", "decode"),
        modes=(
            (
                "edit_image",
                ModularModeTruth(
                    None,
                    frozenset({"prompt", "image"}),
                    ("text_encoder", "vae_encoder", "denoise", "decoder"),
                    _QWEN_ROUTE_IMAGE_TO_OUTPUT_EDGES,
                ),
            ),
            (
                "multi_image_reference_edit",
                ModularModeTruth(
                    None,
                    frozenset({"prompt", "image"}),
                    ("text_encoder", "vae_encoder", "denoise", "decoder"),
                    _QWEN_ROUTE_IMAGE_TO_OUTPUT_EDGES,
                ),
            ),
        ),
    ),
    "QwenImageLayeredModularPipeline": PinnedModularPipelineTruth(
        blocks_class="QwenImageLayeredAutoBlocks",
        fixed_block_sequence=("text_encoder", "vae_encoder", "denoise", "decode"),
        modes=(
            (
                "layer_decomposition",
                ModularModeTruth(
                    None,
                    frozenset({"prompt", "image", "layers"}),
                    ("text_encoder", "vae_encoder", "denoise", "decoder"),
                    _IMAGE_TO_OUTPUT_EDGES,
                ),
            ),
        ),
    ),
    "FluxModularPipeline": PinnedModularPipelineTruth(
        blocks_class="FluxAutoBlocks",
        workflows=(
            _workflow("text2image", "prompt"),
            _workflow("image2image", "image", "prompt"),
        ),
        modes=(
            (
                "text_to_image",
                ModularModeTruth(
                    "text2image",
                    frozenset({"prompt"}),
                    ("text_encoder", "denoise", "decoder"),
                    _TEXT_TO_OUTPUT_EDGES,
                ),
            ),
            (
                "image_to_image",
                ModularModeTruth(
                    "image2image",
                    frozenset({"image", "prompt"}),
                    ("text_encoder", "vae_encoder", "denoise", "decoder"),
                    _IMAGE_TO_OUTPUT_EDGES,
                ),
            ),
        ),
    ),
    "FluxKontextModularPipeline": PinnedModularPipelineTruth(
        blocks_class="FluxKontextAutoBlocks",
        workflows=(
            _workflow("text2image", "prompt"),
            _workflow("image_conditioned", "image", "prompt"),
        ),
        modes=(
            (
                "text_to_image",
                ModularModeTruth(
                    "text2image",
                    frozenset({"prompt"}),
                    ("text_encoder", "denoise", "decoder"),
                    _TEXT_TO_OUTPUT_EDGES,
                ),
            ),
            (
                "edit_image",
                ModularModeTruth(
                    "image_conditioned",
                    frozenset({"image", "prompt"}),
                    ("text_encoder", "vae_encoder", "denoise", "decoder"),
                    _IMAGE_TO_OUTPUT_EDGES,
                ),
            ),
        ),
    ),
    "Flux2KleinModularPipeline": PinnedModularPipelineTruth(
        blocks_class="Flux2KleinAutoBlocks",
        workflows=(
            _workflow("text2image", "prompt"),
            _workflow("image_conditioned", "image", "prompt"),
        ),
        constructor_config=(("is_distilled", True),),
        modes=(
            (
                "text_to_image",
                ModularModeTruth(
                    "text2image",
                    frozenset({"prompt"}),
                    ("text_encoder", "denoise", "decoder"),
                    _TEXT_TO_OUTPUT_EDGES,
                ),
            ),
            (
                "edit_image",
                ModularModeTruth(
                    "image_conditioned",
                    frozenset({"image", "prompt"}),
                    ("text_encoder", "vae_encoder", "denoise", "decoder"),
                    _IMAGE_TO_OUTPUT_EDGES,
                ),
            ),
        ),
    ),
    "Flux2KleinBaseModularPipeline": PinnedModularPipelineTruth(
        blocks_class="Flux2KleinBaseAutoBlocks",
        workflows=(
            _workflow("text2image", "prompt"),
            _workflow("image_conditioned", "image", "prompt"),
        ),
        modes=(
            (
                "text_to_image",
                ModularModeTruth(
                    "text2image",
                    frozenset({"prompt"}),
                    ("text_encoder", "denoise", "decoder"),
                    _TEXT_TO_OUTPUT_EDGES,
                ),
            ),
            (
                "edit_image",
                ModularModeTruth(
                    "image_conditioned",
                    frozenset({"image", "prompt"}),
                    ("text_encoder", "vae_encoder", "denoise", "decoder"),
                    _IMAGE_TO_OUTPUT_EDGES,
                ),
            ),
        ),
    ),
    "ZImageModularPipeline": PinnedModularPipelineTruth(
        blocks_class="ZImageAutoBlocks",
        workflows=(
            _workflow("text2image", "prompt"),
            _workflow("image2image", "image", "prompt"),
        ),
        modes=(
            (
                "text_to_image",
                ModularModeTruth(
                    "text2image",
                    frozenset({"prompt"}),
                    ("text_encoder", "denoise", "decoder"),
                    _TEXT_TO_OUTPUT_EDGES,
                ),
            ),
            (
                "image_to_image",
                ModularModeTruth(
                    "image2image",
                    frozenset({"image", "prompt"}),
                    ("text_encoder", "vae_encoder", "denoise", "decoder"),
                    _IMAGE_TO_OUTPUT_EDGES,
                ),
            ),
        ),
    ),
    "WanModularPipeline": PinnedModularPipelineTruth(
        blocks_class="WanBlocks",
        fixed_block_sequence=("text_encoder", "denoise", "decode"),
        modes=(
            (
                "text_to_video",
                ModularModeTruth(
                    None,
                    frozenset({"prompt"}),
                    ("text_encoder", "denoise", "decoder"),
                    _TEXT_TO_OUTPUT_EDGES,
                ),
            ),
        ),
    ),
    "WanImage2VideoModularPipeline": PinnedModularPipelineTruth(
        blocks_class="WanImage2VideoAutoBlocks",
        workflows=(
            _workflow("flf2v", "image", "last_image", "prompt"),
            _workflow("image2video", "image", "prompt"),
        ),
        modes=(
            (
                "image_to_video",
                ModularModeTruth(
                    "image2video",
                    frozenset({"image", "prompt"}),
                    ("text_encoder", "image_encoder", "vae_encoder", "denoise", "decoder"),
                    _WAN_IMAGE_TO_OUTPUT_EDGES,
                    _WAN_IMAGE2VIDEO_BLOCK_SEQUENCE,
                ),
            ),
        ),
        state_flows=(
            (
                "flf2v",
                ModularStateFlowTruth(
                    "flf2v",
                    frozenset({"image", "last_image", "prompt"}),
                    _WAN_FLF2V_BLOCK_SEQUENCE,
                    ("text_encoder", "image_encoder", "vae_encoder", "denoise", "decoder"),
                    _WAN_IMAGE_TO_OUTPUT_EDGES,
                ),
            ),
        ),
    ),
    "MiniMaxMusic3ModularPipeline": PinnedModularPipelineTruth(
        blocks_class="MiniMaxMusic3Blocks",
        fixed_block_sequence=("semantic_generator", "denoise", "decode"),
        modes=(
            (
                "text_to_audio",
                ModularModeTruth(
                    None,
                    frozenset({"lyrics", "prompt"}),
                    ("semantic_generator", "workflow_denoise", "workflow_audio_decoder"),
                    (
                        StateEdgeTruth(
                            "semantic_generator", "state_out", "workflow_denoise", "state_in"
                        ),
                        StateEdgeTruth(
                            "workflow_denoise", "state_out", "workflow_audio_decoder", "state_in"
                        ),
                    ),
                    ("semantic_generator", "denoise", "decode"),
                ),
            ),
        ),
    ),
}


FLUX_MODULAR_CONTROL_UNSUPPORTED = {
    "status": "unsupported",
    "reason": (
        "Pinned Diffusers Flux Modular blocks provide text-to-image and image-to-image only; no Modular "
        "ControlNet workflow can be assembled."
    ),
    "missingState": ["controlnet_workflow"],
}
