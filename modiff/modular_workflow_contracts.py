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


PINNED_DIFFUSERS_REVISION = "13a7bee4878d62fccc8d25f97e480e68de96fa03"


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

_QWEN_ROUTE_IMAGE_TO_OUTPUT_EDGES = (
    StateEdgeTruth("text_encoder", "embeddings", "denoise", "embeddings"),
    StateEdgeTruth("vae_encoder", "image_latents", "denoise", "image_latents"),
    StateEdgeTruth("vae_encoder", "route_state_out", "denoise", "route_state_in"),
    StateEdgeTruth("denoise", "latents", "decoder", "latents"),
    StateEdgeTruth("denoise", "route_state_out", "decoder", "route_state_in"),
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


PINNED_MODULAR_WORKFLOW_TRUTH: dict[str, PinnedModularPipelineTruth] = {
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
    ),
    "Flux2KleinModularPipeline": PinnedModularPipelineTruth(
        blocks_class="Flux2KleinAutoBlocks",
        workflows=(
            _workflow("text2image", "prompt"),
            _workflow("image_conditioned", "image", "prompt"),
        ),
        constructor_config=(("is_distilled", True),),
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
}


SDXL_MODULAR_INPAINT_UNSUPPORTED = {
    "status": "unsupported",
    "upstreamWorkflow": "inpainting",
    "reason": (
        "MoDiff declares the pinned SDXL base-inpaint encode, denoise, and decode state flow internally, "
        "but no reviewed execution profile or live qualification advertises it as runnable. Combined "
        "ControlNet, ControlNet Union, and IP-Adapter inpaint variants remain outside this base contract."
    ),
    "missingState": [],
}


FLUX_MODULAR_CONTROL_UNSUPPORTED = {
    "status": "unsupported",
    "reason": (
        "Pinned Diffusers Flux Modular blocks provide text-to-image and image-to-image only; no Modular "
        "ControlNet workflow can be assembled."
    ),
    "missingState": ["controlnet_workflow"],
}
