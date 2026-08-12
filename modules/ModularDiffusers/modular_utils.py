# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import logging
import math
import re
import threading
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from diffusers import Flux2KleinModularPipeline
from modiff.model_artifact_catalog import resolve_model_revision
from modiff.modular_workflow_contracts import WAN_I2V_REPOSITORY
from .pipeline_schema import MoDiffParam as PipelineParam
from .pipeline_schema import MoDiffPipelineConfig as PipelineConfig
from .custom_pipeline import (
    CUSTOM_PIPELINE_EXECUTION_STATUS,
    CUSTOM_PIPELINE_IDENTITY_FIELD,
    CUSTOM_PIPELINE_MODEL_TYPE,
    CustomPipelineBinding,
    resolve_custom_pipeline_identity,
)


logger = logging.getLogger("modiff")

_CANONICAL_INTEGER_TEXT = re.compile(r"^(?:0|-?[1-9][0-9]*)$")
_MIN_MODULAR_SEED = 0
_MAX_MODULAR_SEED = 4294967295
SDXL_LAYER_BLOCK_OPTIONS = (
    "down_blocks.1.attentions.0.transformer_blocks",
    "down_blocks.1.attentions.1.transformer_blocks",
    "down_blocks.2.attentions.0.transformer_blocks",
    "down_blocks.2.attentions.1.transformer_blocks",
    "mid_block.attentions.0.transformer_blocks",
    "up_blocks.0.attentions.0.transformer_blocks",
    "up_blocks.0.attentions.1.transformer_blocks",
    "up_blocks.0.attentions.2.transformer_blocks",
    "up_blocks.1.attentions.0.transformer_blocks",
    "up_blocks.1.attentions.1.transformer_blocks",
    "up_blocks.1.attentions.2.transformer_blocks",
)
QWEN_IMAGE_LAYER_BLOCK_OPTIONS = ("transformer_blocks",)
FLUX_LAYER_BLOCK_OPTIONS = ("transformer_blocks", "single_transformer_blocks")
IMAGE_LATENT_DIMENSIONS = ("height", "width")
ALL_GUIDER_OPTIONS = (
    "ClassifierFreeGuidance",
    "SkipLayerGuidance",
    "AdaptiveProjectedGuidance",
    "AdaptiveProjectedMixGuidance",
    "ClassifierFreeZeroStarGuidance",
    "AutoGuidance",
    "SmoothedEnergyGuidance",
    "PerturbedAttentionGuidance",
    "TangentialClassifierFreeGuidance",
    "FrequencyDecoupledGuidance",
)
LAYER_GUIDER_OPTIONS = frozenset(
    {"SkipLayerGuidance", "AutoGuidance", "SmoothedEnergyGuidance", "PerturbedAttentionGuidance"}
)
NON_LAYER_GUIDER_OPTIONS = tuple(name for name in ALL_GUIDER_OPTIONS if name not in LAYER_GUIDER_OPTIONS)
COMPATIBLE_SCHEDULER_OPTIONS = (
    "DDIMScheduler",
    "DDPMScheduler",
    "DEISMultistepScheduler",
    "DPMSolverMultistepScheduler",
    "DPMSolverSinglestepScheduler",
    "DPMSolverSDEScheduler",
    "EulerDiscreteScheduler",
    "EulerAncestralDiscreteScheduler",
    "HeunDiscreteScheduler",
    "KDPM2DiscreteScheduler",
    "KDPM2AncestralDiscreteScheduler",
    "LMSDiscreteScheduler",
    "PNDMScheduler",
    "UniPCMultistepScheduler",
)


def _normalize_modular_integer(value):
    """Accept graph integers without silently truncating another JSON type."""

    if isinstance(value, bool):
        raise ValueError("expected an integer, not a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and _CANONICAL_INTEGER_TEXT.fullmatch(value):
        return int(value)
    raise ValueError("expected an integer or canonical integer string")


def reject_undeclared_modular_generator(kwargs, declared_params=()):
    """Reject graph-supplied Torch generator state outside a declared field contract."""

    if "generator" in kwargs and "generator" not in declared_params:
        raise ValueError(
            "Direct Modular Diffusers 'generator' values are not accepted by this graph action. "
            "Use its backend-issued seed field so MoDiff can construct the generator on the execution device."
        )


def normalize_modular_runtime_params(kwargs, node_config):
    """Cast and enforce scalar constraints from a backend action schema."""

    normalized = dict(kwargs)
    declared_params = node_config.get("params", {})
    reject_undeclared_modular_generator(normalized, declared_params)

    for param_name, param_config in declared_params.items():
        if param_name not in normalized or normalized[param_name] is None:
            continue
        value = normalized[param_name]
        param_type = param_config.get("type")
        try:
            if param_type == "float":
                if isinstance(value, bool):
                    raise ValueError("expected a finite number, not a boolean")
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("expected a finite number")
            elif param_type == "int":
                value = _normalize_modular_integer(value)
            elif param_type == "boolean" and not isinstance(value, bool):
                raise ValueError("expected a JSON boolean")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid value for modular parameter '{param_name}': expected {param_type}.") from exc

        options = param_config.get("options")
        if isinstance(options, list) and value not in options:
            raise ValueError(
                f"Invalid value for modular parameter '{param_name}': expected one of {options}."
            )
        minimum = param_config.get("min")
        maximum = param_config.get("max")
        if minimum is not None and value < minimum:
            raise ValueError(
                f"Invalid value for modular parameter '{param_name}': expected a value greater than or equal to "
                f"{minimum}."
            )
        if maximum is not None and value > maximum:
            raise ValueError(
                f"Invalid value for modular parameter '{param_name}': expected a value less than or equal to "
                f"{maximum}."
            )
        normalized[param_name] = value
    return normalized


def modular_generator_from_seed(seed, pipeline):
    """Build a deterministic Torch generator on a Modular pipeline's execution device."""

    normalized_seed = normalize_modular_seed(seed)

    execution_device = getattr(pipeline, "_execution_device", None)
    if execution_device is None:
        raise RuntimeError(
            "The Modular Diffusers encoder could not resolve its execution device for deterministic sampling."
        )
    return torch.Generator(device=execution_device).manual_seed(normalized_seed)


def normalize_modular_seed(seed):
    """Return one bounded canonical seed without accepting JSON lookalikes."""

    try:
        normalized_seed = _normalize_modular_integer(seed)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid Modular Diffusers seed: expected a finite integer.") from exc
    if not _MIN_MODULAR_SEED <= normalized_seed <= _MAX_MODULAR_SEED:
        raise ValueError(
            f"Invalid Modular Diffusers seed: expected {_MIN_MODULAR_SEED} through {_MAX_MODULAR_SEED}."
        )
    return normalized_seed


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
            PipelineParam.mask(),
            PipelineParam.masked_image_latents(),
            PipelineParam.controlnet_bundle(display="input"),
            PipelineParam.ip_adapter(),
            PipelineParam.route_state_in(),
        ],
        "model_inputs": [
            PipelineParam.unet(),
            PipelineParam.vae(),
            PipelineParam.guider(),
            PipelineParam.scheduler(),
            PipelineParam.controlnet_bundle(display="input"),
        ],
        "outputs": [
            PipelineParam.latents(display="output"),
            PipelineParam.latents_preview(),
            PipelineParam.route_state_out(),
            PipelineParam.doc(),
        ],
        "required_inputs": ["embeddings"],
        "required_model_inputs": ["unet", "vae", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            PipelineParam.image(),
            PipelineParam.mask_image(),
            PipelineParam.padding_mask_crop(),
            PipelineParam.seed(),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.image_latents(display="output"),
            PipelineParam.mask(display="output"),
            PipelineParam.masked_image_latents(display="output"),
            PipelineParam.route_state_out(),
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
            PipelineParam.route_state_in(),
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
    layer_block_options=SDXL_LAYER_BLOCK_OPTIONS,
    guider_options=ALL_GUIDER_OPTIONS,
    scheduler_options=COMPATIBLE_SCHEDULER_OPTIONS,
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
            PipelineParam.seed(),
            PipelineParam.route_state_in(),
        ],
        "model_inputs": [
            PipelineParam.controlnet(),
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.controlnet_bundle(display="output"),
            PipelineParam.route_state_out(),
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
            PipelineParam.route_state_in(),
        ],
        "model_inputs": [
            PipelineParam.unet(),
            PipelineParam.guider(),
            PipelineParam.scheduler(),
            PipelineParam.controlnet_bundle(display="input"),
        ],
        "outputs": [
            PipelineParam.latents(display="output"),
            PipelineParam.route_state_out(),
            PipelineParam.doc(),
        ],
        "required_inputs": ["embeddings"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            PipelineParam.image(),
            PipelineParam.mask_image(),
            PipelineParam.padding_mask_crop(),
            PipelineParam.height(),
            PipelineParam.width(),
            PipelineParam.seed(),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.image_latents(display="output"),
            PipelineParam.route_state_out(),
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
            PipelineParam.route_state_in(),
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
    layer_block_options=QWEN_IMAGE_LAYER_BLOCK_OPTIONS,
    guider_options=ALL_GUIDER_OPTIONS,
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
            PipelineParam.route_state_in(),
        ],
        "model_inputs": [
            PipelineParam.unet(),
            PipelineParam.guider(),
            PipelineParam.scheduler(),
        ],
        "outputs": [
            PipelineParam.latents(display="output"),
            PipelineParam.route_state_out(),
            PipelineParam.doc(),
        ],
        "required_inputs": ["embeddings", "image_latents"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            PipelineParam.image(),
            PipelineParam.mask_image(),
            PipelineParam.padding_mask_crop(),
            PipelineParam.seed(),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.image_latents(display="output"),
            PipelineParam.route_state_out(),
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
            PipelineParam.route_state_in(),
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
    layer_block_options=QWEN_IMAGE_LAYER_BLOCK_OPTIONS,
    guider_options=ALL_GUIDER_OPTIONS,
    denoise_image_latent_dimensions=IMAGE_LATENT_DIMENSIONS,
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
            PipelineParam.route_state_in(),
        ],
        "model_inputs": [
            PipelineParam.unet(),
            PipelineParam.guider(),
            PipelineParam.scheduler(),
        ],
        "outputs": [
            PipelineParam.latents(display="output"),
            PipelineParam.route_state_out(),
            PipelineParam.doc(),
        ],
        "required_inputs": ["embeddings", "image_latents"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            PipelineParam.image(),
            PipelineParam.seed(),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.image_latents(display="output"),
            PipelineParam.route_state_out(),
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
            PipelineParam.route_state_in(),
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
    layer_block_options=QWEN_IMAGE_LAYER_BLOCK_OPTIONS,
    guider_options=ALL_GUIDER_OPTIONS,
    denoise_image_latent_dimensions=IMAGE_LATENT_DIMENSIONS,
)

# =============================================================================
# Qwen Image Layered
# =============================================================================


def _qwen_image_layered_resolution_param():
    """Return the shared, backend-owned Layered source-resolution contract."""

    return PipelineParam(
        name="resolution",
        label="Source Resolution",
        type="int",
        default=640,
        options=[640, 1024],
        fieldOptions={
            "controlTier": "advanced",
            "studioBinding": {
                "schemaVersion": 1,
                "group": "source-resolution",
                "formFields": ["width", "height"],
                "transform": "nearest-option-to-long-edge",
            },
        },
        required_block_params=["resolution"],
    )


def _qwen_image_layered_max_sequence_length_param():
    """Return the Layered prompt-length contract and its generic form binding."""

    return PipelineParam(
        name="max_sequence_length",
        label="Maximum Sequence Length",
        type="int",
        default=1024,
        min=1,
        max=1024,
        step=1,
        fieldOptions={
            "controlTier": "advanced",
            "studioBinding": {
                "schemaVersion": 1,
                "group": "maximum-sequence-length",
                "formFields": ["maxSequenceLength"],
                "transform": "identity",
            },
        },
        required_block_params=["max_sequence_length"],
    )


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
            _qwen_image_layered_resolution_param(),
            PipelineParam.seed(),
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
            _qwen_image_layered_resolution_param(),
            PipelineParam(
                name="use_en_prompt",
                label="Use English Prompt Template",
                type="boolean",
                default=False,
                fieldOptions={"controlTier": "advanced"},
                required_block_params=["use_en_prompt"],
            ),
            _qwen_image_layered_max_sequence_length_param(),
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
    guider_options=NON_LAYER_GUIDER_OPTIONS,
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
            PipelineParam.seed(),
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
    layer_block_options=FLUX_LAYER_BLOCK_OPTIONS,
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
            PipelineParam.seed(),
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
    layer_block_options=FLUX_LAYER_BLOCK_OPTIONS,
    denoise_image_latent_dimensions=IMAGE_LATENT_DIMENSIONS,
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
            PipelineParam.seed(),
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
    denoise_image_latent_dimensions=IMAGE_LATENT_DIMENSIONS,
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
            PipelineParam.seed(),
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
    guider_options=NON_LAYER_GUIDER_OPTIONS,
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
    guider_options=NON_LAYER_GUIDER_OPTIONS,
    scheduler_options=COMPATIBLE_SCHEDULER_OPTIONS,
)

WAN_I2V_NODE_SPECS = {
    "controlnet": None,
    "denoise": {
        "inputs": [
            PipelineParam.embeddings(display="input"),
            PipelineParam.width(832, max=8192),
            PipelineParam.height(480, max=8192),
            PipelineParam.seed(),
            PipelineParam.num_inference_steps(50),
            PipelineParam.guidance_scale(5.0),
            PipelineParam.num_frames(81),
            PipelineParam.image_embeds(display="input"),
            PipelineParam.image_condition_latents(display="input"),
            PipelineParam.route_state_in(),
        ],
        "model_inputs": [
            PipelineParam.unet(),
            # Provenance-only: the pinned split denoise block does not consume
            # VAE weights, but its latent defaults must match the encoder VAE.
            PipelineParam.vae(),
            PipelineParam.guider(),
            PipelineParam.scheduler(),
        ],
        "outputs": [
            PipelineParam.latents(display="output"),
            PipelineParam.route_state_out(),
            PipelineParam.doc(),
        ],
        "required_inputs": ["embeddings", "image_embeds", "image_condition_latents"],
        "required_model_inputs": ["unet", "vae", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            PipelineParam.image(),
            PipelineParam.last_image(),
            PipelineParam.height(480, max=8192),
            PipelineParam.width(832, max=8192),
            PipelineParam.num_frames(81),
            PipelineParam.seed(),
            PipelineParam.route_state_in(),
        ],
        "model_inputs": [
            PipelineParam.vae(),
        ],
        "outputs": [
            PipelineParam.image_condition_latents(),
            PipelineParam.route_state_out(),
            PipelineParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "image_encoder": {
        "inputs": [
            PipelineParam.image(),
            PipelineParam.last_image(),
            PipelineParam.height(480, max=8192),
            PipelineParam.width(832, max=8192),
        ],
        "model_inputs": [
            PipelineParam.image_encoder(),
        ],
        "outputs": [
            PipelineParam.image_embeds(display="output"),
            PipelineParam.route_state_out(),
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
            PipelineParam.route_state_in(),
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
    default_repo=WAN_I2V_REPOSITORY,
    default_dtype="bfloat16",
    guider_options=NON_LAYER_GUIDER_OPTIONS,
    scheduler_options=COMPATIBLE_SCHEDULER_OPTIONS,
    loader_component_outputs=("image_encoder",),
)


class DummyCustomPipeline:
    """Unbound registry marker for the custom pipeline choice."""

    def __new__(cls):
        raise ValueError(
            "The Custom Modular Diffusers choice is contract_only and is not an executable pipeline. "
            "Select a source, repository, immutable Hub revision when applicable, and trust setting so the backend "
            "can issue a verified contract checksum for preview."
        )


def pipeline_class_from_model_type(model_type):
    """Resolve a selected Modular Diffusers model type with an actionable error."""

    if isinstance(model_type, Mapping):
        return resolve_custom_pipeline_identity(model_type)
    normalized_model_type = str(model_type or "").strip()
    if not normalized_model_type:
        return None
    if normalized_model_type == CUSTOM_PIPELINE_MODEL_TYPE:
        raise ValueError(
            "The custom Modular Diffusers selection is missing its backend-issued contract identity. "
            "Refresh the Models Loader after selecting its source, repository, revision, and trust setting."
        )

    import diffusers as diffusers_module

    pipeline_class = getattr(diffusers_module, normalized_model_type, None)
    if pipeline_class is None:
        raise ValueError(
            f"Unknown Diffusers modular pipeline class '{normalized_model_type}'. "
            "Install a Diffusers version that provides this model type, then refresh the node definition."
        )
    return pipeline_class


def pipeline_class_from_runtime_inputs(current_pipeline_class, *runtime_values):
    """Recover a modular pipeline class from self-describing connected inputs.

    Dynamic node-definition signals are transient and are not replayed when the
    runtime recreates node instances between tasks. ModelsLoader therefore adds
    ``model_type`` to each component payload, allowing downstream nodes to
    restore the same pipeline contract from their graph inputs.
    """
    model_types = set()
    custom_identities = []
    custom_markers = 0
    custom_markers_without_identity = 0
    visited = set()
    pending = [(runtime_value, 0) for runtime_value in runtime_values]
    value_count = 0
    while pending:
        value, depth = pending.pop()
        value_count += 1
        if value_count > 4096 or depth > 32:
            raise ValueError("Connected Modular Diffusers inputs exceed the safe nested-value limit.")
        if isinstance(value, dict):
            value_id = id(value)
            if value_id in visited:
                continue
            visited.add(value_id)
            if len(value) > 1024:
                raise ValueError("Connected Modular Diffusers inputs exceed the safe container-size limit.")
            model_type = value.get("model_type")
            if isinstance(model_type, str) and model_type.strip():
                model_types.add(model_type.strip())
                if model_type.strip() == CUSTOM_PIPELINE_MODEL_TYPE:
                    custom_markers += 1
                    identity = value.get(CUSTOM_PIPELINE_IDENTITY_FIELD)
                    if isinstance(identity, Mapping):
                        custom_identities.append(identity)
                    else:
                        custom_markers_without_identity += 1
            pending.extend(
                (nested_value, depth + 1)
                for key, nested_value in value.items()
                if key != CUSTOM_PIPELINE_IDENTITY_FIELD
            )
        elif isinstance(value, (list, tuple)):
            if len(value) > 1024:
                raise ValueError("Connected Modular Diffusers inputs exceed the safe container-size limit.")
            pending.extend((nested_value, depth + 1) for nested_value in value)

    if not model_types:
        return current_pipeline_class
    if len(model_types) > 1:
        raise ValueError(
            "Connected modular model inputs use incompatible pipeline classes: " + ", ".join(sorted(model_types))
        )

    model_type = next(iter(model_types))
    current_model_type = getattr(current_pipeline_class, "__name__", None)
    if current_pipeline_class is not None and current_model_type != model_type:
        selected_name = current_model_type or type(current_pipeline_class).__name__
        raise ValueError(
            f"The Modular Diffusers node is configured for pipeline class '{selected_name}', but its connected "
            f"model inputs identify '{model_type}'. Reconnect components from one Models Loader or update the "
            "node so the selected and connected pipeline classes match."
        )

    if model_type == CUSTOM_PIPELINE_MODEL_TYPE:
        if custom_markers == 0 or custom_markers_without_identity or not custom_identities:
            raise ValueError(
                "Connected custom Modular Diffusers inputs are missing their backend-issued contract identity. "
                "Refresh and rerun the Models Loader before executing this node."
            )
        bindings = [resolve_custom_pipeline_identity(identity) for identity in custom_identities]
        binding = bindings[0]
        if any(resolved.identity != binding.identity for resolved in bindings[1:]):
            raise ValueError("Connected custom Modular Diffusers inputs use incompatible contract identities.")

        if current_pipeline_class is not None:
            if not isinstance(current_pipeline_class, CustomPipelineBinding):
                raise ValueError(
                    "The Modular Diffusers node is configured with an unverified custom pipeline marker. "
                    "Refresh its backend-issued contract identity before running."
                )
            if current_pipeline_class.identity != binding.identity:
                raise ValueError(
                    "The Modular Diffusers node is configured for a different custom contract identity than its "
                    "connected model inputs. Reconnect components from one Models Loader or refresh the node."
                )
            return current_pipeline_class
        return binding

    pipeline_class = pipeline_class_from_model_type(model_type)
    return current_pipeline_class or pipeline_class


DUMMY_CUSTOM_PIPELINE_CONFIG = PipelineConfig(node_specs={}, label="Custom", default_repo="", default_dtype="bfloat16")


# Minimal modular registry for MoDiff node configs
class ModiffPipelineRegistry:
    """Registry mapping pipeline class to its config, including label, default_repo, default_dtype, and node_params."""

    def __init__(self):
        self._registry: Dict[type, PipelineConfig] = {}
        self._initialized = False
        # Registration occurs while initialization already owns this lock.
        self._init_lock = threading.RLock()

    def register(self, pipeline_cls: type, config: PipelineConfig):
        """Register a pipeline class with its config."""
        with self._init_lock:
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
            return dict(self._registry)


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

    Returns model_type, label, default_repo, default_dtype, and node_params.
    The custom preview marker also declares ``execution_status=contract_only``.
    """
    registry = _get_registry_instance().get_all()
    for pipeline_cls, config in registry.items():
        if pipeline_cls.__name__ == model_type:
            metadata = {
                "model_type": model_type,
                "label": config.label,
                "default_repo": config.default_repo,
                "default_dtype": config.default_dtype,
                "loader_component_outputs": list(config.loader_component_outputs),
                "layer_block_options": list(config.layer_block_options),
                "guider_options": list(config.guider_options),
                "scheduler_options": list(config.scheduler_options),
                "denoise_image_latent_dimensions": list(config.denoise_image_latent_dimensions),
                "node_params": config.node_params,
            }
            if model_type == CUSTOM_PIPELINE_MODEL_TYPE:
                metadata["execution_status"] = CUSTOM_PIPELINE_EXECUTION_STATUS
            return metadata
    return None


def get_modular_layer_block_options() -> Dict[str, list[str]]:
    """Return exact model-type-to-transformer-block choices from reviewed configs."""

    return {
        pipeline_cls.__name__: list(config.layer_block_options)
        for pipeline_cls, config in _get_registry_instance().get_all().items()
        if config.layer_block_options
    }


def get_modular_guider_options() -> Dict[str, list[str]]:
    """Return exact model-type-to-guider choices from reviewed configs."""

    return {
        pipeline_cls.__name__: list(config.guider_options)
        for pipeline_cls, config in _get_registry_instance().get_all().items()
        if config.guider_options
    }


def get_modular_scheduler_options() -> Dict[str, list[str]]:
    """Return exact model-type-to-scheduler replacements from reviewed configs."""

    return {
        pipeline_cls.__name__: list(config.scheduler_options)
        for pipeline_cls, config in _get_registry_instance().get_all().items()
        if config.scheduler_options
    }


def pipeline_class_to_modiff_node_config(pipeline_class, node_type=None, *, resolve_blocks=True):
    """Get the block and MoDiff node parameters for a pipeline class and node type."""
    if isinstance(pipeline_class, CustomPipelineBinding):
        config = pipeline_class.pipeline_config()
    else:
        config = _get_registry_instance().get(pipeline_class)
    if config is None:
        logger.debug(f"Failed to load config for {pipeline_class}")
        return None, None

    node_params = config.node_params.get(node_type)

    node_type_blocks = None
    if resolve_blocks and node_params is not None and node_params.get("block_name"):
        # patch to use only distilled klein blocks
        if pipeline_class == Flux2KleinModularPipeline:
            pipeline = pipeline_class(config_dict={"is_distilled": True})
        else:
            pipeline = pipeline_class()

        node_type_blocks = pipeline.blocks.sub_blocks[node_params["block_name"]]

    return node_type_blocks, node_params


_MODIFF_NODE_ACTION_LABELS = {
    "text_encoder": "Encode Prompt",
    "image_encoder": "Image Embeddings",
    "vae_encoder": "Encode Image",
    "denoise": "Denoise",
    "decoder": "Decode Latents",
    "controlnet": "ControlNet",
}


def require_modiff_node_contract(pipeline_class, node_type, *, require_blocks=True, resolve_blocks=True):
    """Resolve an isolated Modular node-action contract or fail before execution.

    Registry-backed custom configurations retain their deserialized ``node_params``
    dictionary. Dynamic node definitions remove their own connector from the
    returned parameter set, so every caller must receive a deep copy rather than
    the registry-owned object. Bundle-only actions such as SDXL ControlNet may
    opt out of executable-block validation with ``require_blocks=False``.
    UI-only refreshes use ``resolve_blocks=False`` so selecting a custom
    contract cannot import repository Python before graph execution.
    """

    action_label = _MODIFF_NODE_ACTION_LABELS.get(node_type, str(node_type).replace("_", " ").title())
    if pipeline_class is None:
        raise ValueError(
            f"The generic Modular Diffusers {action_label} node requires connected model inputs that identify a "
            "supported Modular pipeline. Reconnect the Models Loader output and update the node before running."
        )

    blocks, node_config = pipeline_class_to_modiff_node_config(
        pipeline_class,
        node_type,
        resolve_blocks=resolve_blocks,
    )
    pipeline_name = getattr(pipeline_class, "__name__", type(pipeline_class).__name__)
    if node_config is None:
        raise ValueError(
            f"Modular Diffusers pipeline '{pipeline_name}' does not support the generic {action_label} node "
            f"(action '{node_type}'). Select a pipeline with a registered {action_label} contract or remove the "
            "unsupported node from this workflow."
        )
    if resolve_blocks and require_blocks and blocks is None:
        raise ValueError(
            f"Modular Diffusers pipeline '{pipeline_name}' has an incomplete {action_label} contract "
            f"(action '{node_type}'): its registered block could not be resolved. Repair the pipeline config "
            "before running this workflow."
        )
    return blocks, deepcopy(node_config)
