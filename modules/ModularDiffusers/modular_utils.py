# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from diffusers import Flux2KleinModularPipeline
from modiff.model_artifact_catalog import resolve_model_revision
from .pipeline_schema import MoDiffParam as PipelineParam
from .pipeline_schema import MoDiffPipelineConfig as PipelineConfig


logger = logging.getLogger("modiff")

IMMUTABLE_HUB_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")


def require_immutable_hub_revision(repo_id, revision, *, required: bool):
    """Require a commit hash before loading a trust-sensitive Hub repository."""

    repository = str(repo_id or "").strip()
    normalized_revision = str(revision or "").strip() or None
    if not repository or Path(repository).expanduser().exists():
        return normalized_revision
    if required and not (normalized_revision and IMMUTABLE_HUB_REVISION.fullmatch(normalized_revision)):
        raise ValueError(
            "Remote Modular Diffusers repositories require an immutable 40-character Hugging Face commit revision. "
            "Review the repository contents, copy its commit hash into Revision, and retry."
        )
    return normalized_revision


def pin_modular_component_revisions(pipeline, primary_repo, primary_revision):
    """Attach resolved Hub revisions to the component specs a modular config creates.

    Upstream ``ModularPipeline.from_pretrained`` uses ``revision`` to fetch the
    pipeline configuration, but the current experimental API does not copy it
    into the component specs built from that configuration. Without this pass,
    ``load_components`` can still follow an auxiliary repository's moving
    default branch even though the graph and top-level config were pinned.
    Unknown remote auxiliaries fail before any component spec is mutated.
    """

    primary = str(primary_repo or "").strip()
    resolved_specs = []
    for name, spec in (getattr(pipeline, "_component_specs", None) or {}).items():
        repository = getattr(spec, "pretrained_model_name_or_path", None)
        if not isinstance(repository, str) or not repository.strip():
            continue
        if primary and repository.lower() == primary.lower():
            revision = primary_revision
            if not revision:
                continue
        else:
            revision = resolve_model_revision(repository, getattr(spec, "revision", None))
            revision = require_immutable_hub_revision(repository, revision, required=True)
            if not revision:
                continue
        resolved_specs.append((str(name), spec, revision))

    applied = {}
    for name, spec, revision in resolved_specs:
        spec.revision = revision
        applied[name] = revision
    return applied


SDXL_NODE_SPECS = {
    "controlnet": {
        "inputs": [
            PipelineParam.control_image(),
            PipelineParam.controlnet_conditioning_scale(),
            PipelineParam.control_guidance_start(),
            PipelineParam.control_guidance_end(),
            PipelineParam.height(),
            PipelineParam.width(),
        ],
        "model_inputs": [
            PipelineParam.controlnet(),
        ],
        "outputs": [
            PipelineParam.controlnet_bundle(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["control_image"],
        "required_model_inputs": ["controlnet"],
        "block_name": None,
    },
    "denoise": {
        "inputs": [
            PipelineParam.embeddings(display="input"),
            PipelineParam.width(),
            PipelineParam.height(),
            PipelineParam.seed(),
            PipelineParam.num_inference_steps(),
            PipelineParam.guidance_scale(),
            PipelineParam.image_latents_with_strength(),
            PipelineParam.strength(),
            PipelineParam.controlnet_bundle(display="input"),
            PipelineParam.ip_adapter(),
        ],
        "model_inputs": [
            PipelineParam.unet(),
            PipelineParam.guider(),
            PipelineParam.scheduler(),
            PipelineParam.controlnet_bundle(display="input"),
        ],
        "outputs": [
            PipelineParam.latents(display="output"),
            PipelineParam.latents_preview(),
            PipelineParam.doc(),
        ],
        "required_inputs": ["embeddings"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            PipelineParam.image(),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.image_latents(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "text_encoder": {
        "inputs": [
            PipelineParam.prompt(),
            PipelineParam.negative_prompt(),
        ],
        "model_inputs": [
            PipelineParam.text_encoders(),
        ],
        "outputs": [
            PipelineParam.embeddings(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["prompt"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            PipelineParam.latents(display="input"),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.images(),
            PipelineParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}

SDXL_PIPELINE_CONFIG = PipelineConfig(
    node_specs=SDXL_NODE_SPECS,
    label="Stable Diffusion XL",
    default_repo="stabilityai/stable-diffusion-xl-base-1.0",
    default_dtype="float16",
)


# =============================================================================
# Qwen Image
# =============================================================================

QWEN_IMAGE_NODE_SPECS = {
    "controlnet": {
        "inputs": [
            PipelineParam.control_image(),
            PipelineParam.controlnet_conditioning_scale(),
            PipelineParam.control_guidance_start(),
            PipelineParam.control_guidance_end(),
            PipelineParam.height(),
            PipelineParam.width(),
        ],
        "model_inputs": [
            PipelineParam.controlnet(),
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.controlnet_bundle(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["control_image"],
        "required_model_inputs": ["controlnet", "vae"],
        "block_name": "controlnet_vae_encoder",
    },
    "denoise": {
        "inputs": [
            PipelineParam.embeddings(display="input"),
            PipelineParam.width(),
            PipelineParam.height(),
            PipelineParam.seed(),
            PipelineParam.num_inference_steps(50),
            PipelineParam.guidance_scale(4.5),
            PipelineParam.image_latents_with_strength(),
            PipelineParam.strength(),
            PipelineParam.controlnet_bundle(display="input"),
        ],
        "model_inputs": [
            PipelineParam.unet(),
            PipelineParam.guider(),
            PipelineParam.scheduler(),
            PipelineParam.controlnet_bundle(display="input"),
        ],
        "outputs": [
            PipelineParam.latents(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["embeddings"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            PipelineParam.image(),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.image_latents(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "text_encoder": {
        "inputs": [
            PipelineParam.prompt(),
            PipelineParam.negative_prompt(),
        ],
        "model_inputs": [
            PipelineParam.text_encoders(),
        ],
        "outputs": [
            PipelineParam.embeddings(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["prompt"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            PipelineParam.latents(display="input"),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.images(),
            PipelineParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}

QWEN_IMAGE_PIPELINE_CONFIG = PipelineConfig(
    node_specs=QWEN_IMAGE_NODE_SPECS,
    label="Qwen-Image-2512",
    default_repo="Qwen/Qwen-Image-2512",
    default_dtype="bfloat16",
)


# =============================================================================
# Qwen Image Edit
# =============================================================================

QWEN_IMAGE_EDIT_NODE_SPECS = {
    "controlnet": None,
    "denoise": {
        "inputs": [
            PipelineParam.embeddings(display="input"),
            PipelineParam.seed(),
            PipelineParam.num_inference_steps(40),
            PipelineParam.guidance_scale(4.0),
            PipelineParam.image_latents(display="input"),
        ],
        "model_inputs": [
            PipelineParam.unet(),
            PipelineParam.guider(),
            PipelineParam.scheduler(),
        ],
        "outputs": [
            PipelineParam.latents(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["embeddings", "image_latents"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            PipelineParam.image(),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.image_latents(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "text_encoder": {
        "inputs": [
            PipelineParam.prompt(),
            PipelineParam.negative_prompt(),
            PipelineParam.image(),
        ],
        "model_inputs": [
            PipelineParam.text_encoders(),
        ],
        "outputs": [
            PipelineParam.embeddings(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["prompt", "image"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            PipelineParam.latents(display="input"),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.images(),
            PipelineParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}

QWEN_IMAGE_EDIT_PIPELINE_CONFIG = PipelineConfig(
    node_specs=QWEN_IMAGE_EDIT_NODE_SPECS,
    label="Qwen-Image-Edit",
    default_repo="Qwen/Qwen-Image-Edit",
    default_dtype="bfloat16",
)


# =============================================================================
# Qwen Image Edit Plus
# =============================================================================

QWEN_IMAGE_EDIT_PLUS_NODE_SPECS = {
    "controlnet": None,
    "denoise": {
        "inputs": [
            PipelineParam.embeddings(display="input"),
            PipelineParam.seed(),
            PipelineParam.num_inference_steps(40),
            PipelineParam.guidance_scale(4.0),
            PipelineParam.image_latents(display="input"),
        ],
        "model_inputs": [
            PipelineParam.unet(),
            PipelineParam.guider(),
            PipelineParam.scheduler(),
        ],
        "outputs": [
            PipelineParam.latents(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["embeddings", "image_latents"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            PipelineParam.image(),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.image_latents(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "text_encoder": {
        "inputs": [
            PipelineParam.prompt(),
            PipelineParam.negative_prompt(),
            PipelineParam.image(),
        ],
        "model_inputs": [
            PipelineParam.text_encoders(),
        ],
        "outputs": [
            PipelineParam.embeddings(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["prompt", "image"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            PipelineParam.latents(display="input"),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.images(),
            PipelineParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}

QWEN_IMAGE_EDIT_PLUS_PIPELINE_CONFIG = PipelineConfig(
    node_specs=QWEN_IMAGE_EDIT_PLUS_NODE_SPECS,
    label="Qwen-Image-Edit-2511",
    default_repo="Qwen/Qwen-Image-Edit-2511",
    default_dtype="bfloat16",
)

# =============================================================================
# Qwen Image Layered
# =============================================================================

QWEN_IMAGE_LAYERED_NODE_SPECS = {
    "controlnet": None,
    "denoise": {
        "inputs": [
            PipelineParam.embeddings(display="input"),
            PipelineParam.seed(),
            PipelineParam.num_inference_steps(50),
            PipelineParam.guidance_scale(4.0),
            PipelineParam.layers(4),
            PipelineParam.image_latents(display="input"),
        ],
        "model_inputs": [
            PipelineParam.unet(),
            PipelineParam.guider(),
            PipelineParam.scheduler(),
        ],
        "outputs": [
            PipelineParam.latents(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["embeddings", "image_latents"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            PipelineParam.image(),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.image_latents(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "text_encoder": {
        "inputs": [
            PipelineParam.prompt(),
            PipelineParam.negative_prompt(),
            PipelineParam.image(),
        ],
        "model_inputs": [
            PipelineParam.text_encoders(),
        ],
        "outputs": [
            PipelineParam.embeddings(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["prompt", "image"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            PipelineParam.latents(display="input"),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.images(),
            PipelineParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}

QWEN_IMAGE_LAYERED_PIPELINE_CONFIG = PipelineConfig(
    node_specs=QWEN_IMAGE_LAYERED_NODE_SPECS,
    label="Qwen-Image-Layered",
    default_repo="Qwen/Qwen-Image-Layered",
    default_dtype="bfloat16",
)

# =============================================================================
# Flux
# =============================================================================

FLUX_NODE_SPECS = {
    "controlnet": None,  # Not yet supported in Modular
    "denoise": {
        "inputs": [
            PipelineParam.embeddings(display="input"),
            PipelineParam.width(),
            PipelineParam.height(),
            PipelineParam.seed(),
            PipelineParam.num_inference_steps(28),
            PipelineParam.guidance_scale(3.5),
            PipelineParam.image_latents_with_strength(),
            PipelineParam.strength(),
        ],
        "model_inputs": [
            PipelineParam.unet(),
            PipelineParam.guider(),
            PipelineParam.scheduler(),
        ],
        "outputs": [
            PipelineParam.latents(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["embeddings"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            PipelineParam.image(),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.image_latents(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "text_encoder": {
        "inputs": [
            PipelineParam.prompt(),
            # No negative_prompt - pipeline does not support this
        ],
        "model_inputs": [
            PipelineParam.text_encoders(),
        ],
        "outputs": [
            PipelineParam.embeddings(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["prompt"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            PipelineParam.latents(display="input"),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.images(),
            PipelineParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}

FLUX_PIPELINE_CONFIG = PipelineConfig(
    node_specs=FLUX_NODE_SPECS,
    label="Flux",
    default_repo="black-forest-labs/FLUX.1-dev",
    default_dtype="bfloat16",
)


# =============================================================================
# Flux Kontext
# =============================================================================

FLUX_KONTEXT_NODE_SPECS = {
    "controlnet": None,
    "denoise": {
        "inputs": [
            PipelineParam.embeddings(display="input"),
            PipelineParam.seed(),
            PipelineParam.num_inference_steps(28),
            PipelineParam.guidance_scale(2.5),
            PipelineParam.image_latents(display="input"),
        ],
        "model_inputs": [
            PipelineParam.unet(),
            PipelineParam.guider(),
            PipelineParam.scheduler(),
        ],
        "outputs": [
            PipelineParam.latents(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["embeddings", "image_latents"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            PipelineParam.image(),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.image_latents(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "text_encoder": {
        "inputs": [
            PipelineParam.prompt(),
            # No negative_prompt
        ],
        "model_inputs": [
            PipelineParam.text_encoders(),
        ],
        "outputs": [
            PipelineParam.embeddings(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["prompt"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            PipelineParam.latents(display="input"),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.images(),
            PipelineParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}

FLUX_KONTEXT_PIPELINE_CONFIG = PipelineConfig(
    node_specs=FLUX_KONTEXT_NODE_SPECS,
    label="Flux Kontext",
    default_repo="black-forest-labs/FLUX.1-Kontext-dev",
    default_dtype="bfloat16",
)

# =============================================================================
# Flux 2 Klein
# =============================================================================

FLUX_2_KLEIN_DISTILLED_NODE_SPECS = {
    "controlnet": None,
    "denoise": {
        "inputs": [
            PipelineParam.embeddings(display="input"),
            PipelineParam.width(),
            PipelineParam.height(),
            PipelineParam.seed(),
            PipelineParam.num_inference_steps(4),
            PipelineParam.guidance_scale(1.0),
            PipelineParam.image_latents(display="input"),
        ],
        "model_inputs": [
            PipelineParam.unet(),
            PipelineParam.guider(),
            PipelineParam.scheduler(),
        ],
        "outputs": [
            PipelineParam.latents(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["embeddings"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            PipelineParam.image(),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.image_latents(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "text_encoder": {
        "inputs": [
            PipelineParam.prompt(),
        ],
        "model_inputs": [
            PipelineParam.text_encoders(),
        ],
        "outputs": [
            PipelineParam.embeddings(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["prompt"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            PipelineParam.latents(display="input"),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.images(),
            PipelineParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}

FLUX_2_KLEIN_DISTILLED_PIPELINE_CONFIG = PipelineConfig(
    node_specs=FLUX_2_KLEIN_DISTILLED_NODE_SPECS,
    label="Flux 2 Klein Distilled",
    default_repo="black-forest-labs/FLUX.2-klein-4B",
    default_dtype="bfloat16",
)


# =============================================================================
# Z-Image
# =============================================================================

Z_IMAGE_NODE_SPECS = {
    "controlnet": None,
    "denoise": {
        "inputs": [
            PipelineParam.embeddings(display="input"),
            PipelineParam.width(),
            PipelineParam.height(),
            PipelineParam.seed(),
            PipelineParam.num_inference_steps(9),
            PipelineParam.guidance_scale(1.0),
            PipelineParam.image_latents_with_strength(),
            PipelineParam.strength(),
        ],
        "model_inputs": [
            PipelineParam.unet(),
            PipelineParam.guider(),
            PipelineParam.scheduler(),
        ],
        "outputs": [
            PipelineParam.latents(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["embeddings"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            PipelineParam.image(),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.image_latents(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "text_encoder": {
        "inputs": [
            PipelineParam.prompt(),
            # No negative_prompt - pipeline does not support this
        ],
        "model_inputs": [
            PipelineParam.text_encoders(),
        ],
        "outputs": [
            PipelineParam.embeddings(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["prompt"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            PipelineParam.latents(display="input"),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.images(),
            PipelineParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}

Z_IMAGE_PIPELINE_CONFIG = PipelineConfig(
    node_specs=Z_IMAGE_NODE_SPECS,
    label="Z-Image",
    default_repo="Tongyi-MAI/Z-Image-Turbo",
    default_dtype="bfloat16",
)

# =============================================================================
# WAN
# =============================================================================

WAN_T2V_NODE_SPECS = {
    "controlnet": None,
    "denoise": {
        "inputs": [
            PipelineParam.embeddings(display="input"),
            PipelineParam.width(832),
            PipelineParam.height(480),
            PipelineParam.seed(),
            PipelineParam.num_inference_steps(50),
            PipelineParam.guidance_scale(5.0),
            PipelineParam.num_frames(81),
        ],
        "model_inputs": [
            PipelineParam.unet(),
            PipelineParam.scheduler(),
        ],
        "outputs": [
            PipelineParam.latents(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["embeddings"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "text_encoder": {
        "inputs": [
            PipelineParam.prompt(),
            PipelineParam.negative_prompt(),
        ],
        "model_inputs": [
            PipelineParam.text_encoders(),
        ],
        "outputs": [
            PipelineParam.embeddings(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["prompt"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            PipelineParam.latents(display="input"),
            PipelineParam.output_type(default="pil"),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.videos(),
            PipelineParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}

WAN_T2V_PIPELINE_CONFIG = PipelineConfig(
    node_specs=WAN_T2V_NODE_SPECS,
    label="WAN2 T2V",
    default_repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    default_dtype="bfloat16",
)

WAN_I2V_NODE_SPECS = {
    "controlnet": None,
    "denoise": {
        "inputs": [
            PipelineParam.embeddings(display="input"),
            PipelineParam.width(832),
            PipelineParam.height(480),
            PipelineParam.seed(),
            PipelineParam.num_inference_steps(50),
            PipelineParam.guidance_scale(5.0),
            PipelineParam.num_frames(81),
            PipelineParam.image_embeds(display="input"),
            PipelineParam(name="image_condition_latents", label="Image Latents", type="latents", display="input"),
        ],
        "model_inputs": [
            PipelineParam.unet(),
            PipelineParam.scheduler(),
        ],
        "outputs": [
            PipelineParam.latents(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["embeddings", "image_embeds", "image_condition_latents"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            PipelineParam.image(),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam(name="image_condition_latents", label="Image Latents", type="latents", display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "image_encoder": {
        "inputs": [
            PipelineParam.image(),
        ],
        "model_inputs": [
            PipelineParam.image_encoder(),
        ],
        "outputs": [
            PipelineParam.image_embeds(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["image_encoder"],
        "block_name": "image_encoder",
    },
    "text_encoder": {
        "inputs": [
            PipelineParam.prompt(),
            PipelineParam.negative_prompt(),
        ],
        "model_inputs": [
            PipelineParam.text_encoders(),
        ],
        "outputs": [
            PipelineParam.embeddings(display="output"),
            PipelineParam.doc(),
        ],
        "required_inputs": ["prompt"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            PipelineParam.latents(display="input"),
            PipelineParam(
                name="output_type", label="Output Type", type="dropdown", options=["np", "pil"], default="pil"
            ),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.videos(),
            PipelineParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}

WAN_I2V_PIPELINE_CONFIG = PipelineConfig(
    node_specs=WAN_I2V_NODE_SPECS,
    label="WAN2 I2V",
    default_repo="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    default_dtype="bfloat16",
)


class DummyCustomPipeline:
    """Placeholder class used as registry key for custom pipelines."""

    repo_id = None
    revision = None
    trust_remote_code = False

    def __new__(cls):
        from diffusers import ModularPipeline

        revision = require_immutable_hub_revision(
            cls.repo_id,
            cls.revision,
            required=bool(cls.trust_remote_code),
        )
        kwargs = {
            "trust_remote_code": bool(cls.trust_remote_code),
            "local_files_only": True,
        }
        if revision:
            kwargs["revision"] = revision
        return ModularPipeline.from_pretrained(cls.repo_id, **kwargs)


def pipeline_class_from_runtime_inputs(current_pipeline_class, *runtime_values):
    """Recover a modular pipeline class from self-describing connected inputs.

    Dynamic node-definition signals are transient and are not replayed when the
    runtime recreates node instances between tasks. ModelsLoader therefore adds
    ``model_type`` to each component payload, allowing downstream nodes to
    restore the same pipeline contract from their graph inputs.
    """
    if current_pipeline_class is not None:
        return current_pipeline_class

    model_types = set()
    custom_repositories = set()
    custom_revisions = set()
    custom_trust_values = set()
    visited = set()

    def collect(value):
        if isinstance(value, dict):
            value_id = id(value)
            if value_id in visited:
                return
            visited.add(value_id)
            model_type = value.get("model_type")
            if isinstance(model_type, str) and model_type.strip():
                model_types.add(model_type.strip())
                if model_type.strip() == "DummyCustomPipeline":
                    repository = value.get("repo_id")
                    revision = value.get("revision")
                    if isinstance(repository, str) and repository.strip():
                        custom_repositories.add(repository.strip())
                    if isinstance(revision, str) and revision.strip():
                        custom_revisions.add(revision.strip())
                    custom_trust_values.add(bool(value.get("trust_remote_code")))
            for nested_value in value.values():
                collect(nested_value)
        elif isinstance(value, (list, tuple)):
            for nested_value in value:
                collect(nested_value)

    for runtime_value in runtime_values:
        collect(runtime_value)

    if not model_types:
        return None
    if len(model_types) > 1:
        raise ValueError(
            "Connected modular model inputs use incompatible pipeline classes: " + ", ".join(sorted(model_types))
        )

    model_type = next(iter(model_types))
    if model_type == "DummyCustomPipeline":
        if len(custom_repositories) != 1 or len(custom_revisions) != 1 or len(custom_trust_values) != 1:
            raise ValueError(
                "Connected custom Modular Diffusers inputs have incomplete or conflicting trust metadata."
            )
        repository = next(iter(custom_repositories))
        revision = next(iter(custom_revisions))
        trust_remote_code = next(iter(custom_trust_values))
        require_immutable_hub_revision(repository, revision, required=True)
        DummyCustomPipeline.repo_id = repository
        DummyCustomPipeline.revision = revision
        DummyCustomPipeline.trust_remote_code = trust_remote_code
        return DummyCustomPipeline

    import diffusers as diffusers_module

    pipeline_class = getattr(diffusers_module, model_type, None)
    if pipeline_class is None:
        raise ValueError(
            f"Unknown Diffusers modular pipeline class '{model_type}'. "
            "Install a Diffusers version that provides this model type."
        )
    return pipeline_class


DUMMY_CUSTOM_PIPELINE_CONFIG = PipelineConfig(node_specs={}, label="Custom", default_repo="", default_dtype="bfloat16")


# Minimal modular registry for MoDiff node configs
class ModiffPipelineRegistry:
    """Registry mapping pipeline class to its config, including label, default_repo, default_dtype, and node_params."""

    def __init__(self):
        self._registry: Dict[type, PipelineConfig] = {}
        self._initialized = False
        # Lock to prevent concurrent initialization races
        self._init_lock = threading.Lock()

    def register(self, pipeline_cls: type, config: PipelineConfig):
        """Register a pipeline class with its config."""
        self._registry[pipeline_cls] = config

    def get(self, pipeline_cls: type) -> Optional[PipelineConfig]:
        # Ensure only one thread/coroutine initializes the registry
        with self._init_lock:
            if not self._initialized:
                _initialize_registry(self)
        return self._registry.get(pipeline_cls, None)

    def get_all(self) -> Dict[type, PipelineConfig]:
        # Ensure only one thread/coroutine initializes the registry
        with self._init_lock:
            if not self._initialized:
                _initialize_registry(self)
        return self._registry


def _initialize_registry(registry: ModiffPipelineRegistry):
    """Initialize the registry and register all available pipeline configs."""
    logger.info("Initializing Modular Diffusers registry")

    # register DummyCustomPipeline with empty node specs
    registry.register(DummyCustomPipeline, DUMMY_CUSTOM_PIPELINE_CONFIG)

    try:
        from diffusers import StableDiffusionXLModularPipeline

        registry.register(StableDiffusionXLModularPipeline, SDXL_PIPELINE_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to register StableDiffusionXLModularPipeline: {e}")

    try:
        from diffusers import QwenImageModularPipeline

        registry.register(QwenImageModularPipeline, QWEN_IMAGE_PIPELINE_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to register QwenImageModularPipeline: {e}")

    try:
        from diffusers import QwenImageEditModularPipeline

        registry.register(QwenImageEditModularPipeline, QWEN_IMAGE_EDIT_PIPELINE_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to register QwenImageEditModularPipeline: {e}")

    try:
        from diffusers import QwenImageEditPlusModularPipeline

        registry.register(QwenImageEditPlusModularPipeline, QWEN_IMAGE_EDIT_PLUS_PIPELINE_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to register QwenImageEditPlusModularPipeline: {e}")

    try:
        from diffusers import QwenImageLayeredModularPipeline

        registry.register(QwenImageLayeredModularPipeline, QWEN_IMAGE_LAYERED_PIPELINE_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to register QwenImageLayeredModularPipeline: {e}")

    try:
        from diffusers import FluxModularPipeline

        registry.register(FluxModularPipeline, FLUX_PIPELINE_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to register FluxModularPipeline: {e}")

    try:
        from diffusers import FluxKontextModularPipeline

        registry.register(FluxKontextModularPipeline, FLUX_KONTEXT_PIPELINE_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to register FluxKontextModularPipeline: {e}")

    try:
        from diffusers import Flux2KleinModularPipeline

        registry.register(Flux2KleinModularPipeline, FLUX_2_KLEIN_DISTILLED_PIPELINE_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to register Flux2KleinModularPipeline: {e}")

    try:
        from diffusers import ZImageModularPipeline

        registry.register(ZImageModularPipeline, Z_IMAGE_PIPELINE_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to register ZImageModularPipeline: {e}")

    try:
        from diffusers import WanModularPipeline

        registry.register(WanModularPipeline, WAN_T2V_PIPELINE_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to register WanModularPipeline: {e}")

    try:
        from diffusers import WanImage2VideoModularPipeline

        registry.register(WanImage2VideoModularPipeline, WAN_I2V_PIPELINE_CONFIG)
    except Exception as e:
        logger.warning(f"Failed to register WanImage2VideoModularPipeline: {e}")

    registry._initialized = True


def _get_registry_instance():
    """Lazily import MODULAR_REGISTRY to avoid circular imports"""
    try:
        from . import MODULAR_REGISTRY
    except Exception as e:
        raise RuntimeError("MODULAR_REGISTRY not initialized.") from e
    return MODULAR_REGISTRY


def get_all_model_types() -> Dict[str, str]:
    """Get all registered model types with their labels for UI dropdowns.

    Returns:
        Dict mapping model type names (keys) to human-readable labels (values).

    Example output:
        {
            "": "",
            "StableDiffusionXLModularPipeline": "Stable Diffusion XL",
            "QwenImageModularPipeline": "Qwen Image",
            "FluxModularPipeline": "Flux",
        }
    """
    registry = _get_registry_instance().get_all()
    all_labels = {"": ""}
    for pipeline_cls, config in registry.items():
        model_type = pipeline_cls.__name__
        all_labels[model_type] = config.label
    return all_labels


def get_model_type_metadata(model_type: str) -> Optional[Dict[str, Any]]:
    """Get metadata for a model type.

    Returns dict with model_type, label, default_repo, default_dtype, node_params.
    """
    registry = _get_registry_instance().get_all()
    for pipeline_cls, config in registry.items():
        if pipeline_cls.__name__ == model_type:
            return {
                "model_type": model_type,
                "label": config.label,
                "default_repo": config.default_repo,
                "default_dtype": config.default_dtype,
                "node_params": config.node_params,
            }
    return None


def pipeline_class_to_modiff_node_config(pipeline_class, node_type=None):
    """Get the block and MoDiff node parameters for a pipeline class and node type."""
    config = _get_registry_instance().get(pipeline_class)
    if config is None:
        logger.debug(f"Failed to load config for {pipeline_class}")
        return None, None

    node_params = config.node_params.get(node_type)

    node_type_blocks = None
    if node_params is not None and node_params.get("block_name"):
        # patch to use only distilled klein blocks
        if pipeline_class == Flux2KleinModularPipeline:
            pipeline = pipeline_class(config_dict={"is_distilled": True})
        else:
            pipeline = pipeline_class()

        node_type_blocks = pipeline.blocks.sub_blocks[node_params["block_name"]]

    return node_type_blocks, node_params
