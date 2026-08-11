"""Model-neutral Diffusers video facade nodes.

Node identity describes the media contract; registered adapters own pipeline
loading, input validation, and family-specific argument translation.  Legacy
Wan node keys remain registered separately for persisted workflows.
"""

from dataclasses import dataclass
from functools import wraps
import json
import logging
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from modiff.config import CONFIG
from modiff.NodeBase import NodeBase
from modiff.diffusers_offload import OFFLOAD_MODE_MODEL_CPU, apply_pipeline_offload
from modiff.model_artifact_catalog import IMMUTABLE_HUB_REVISION, catalog_revision, require_catalog_revision
from modules.DiffusersVideo.wan_vace import (
    WAN_VACE_NATIVE_CHUNK_FRAMES,
    WanVACEGenerate,
    WanVACELoadPipeline,
    callback_tensor_inputs,
    ensure_reference_images,
    ensure_single_prompt,
    ensure_video_list,
    none_if_blank,
    normalize_num_frames,
    parse_json_object,
    repo_value,
    validate_dimensions,
)
from utils.huggingface import local_files_only, validate_hf_repo_id
from utils.torch_utils import DEFAULT_DEVICE, str_to_dtype

logger = logging.getLogger("modiff")

# LTX's distilled 13B release is trained for this non-uniform eight-evaluation
# trajectory.  Diffusers otherwise derives a generic linear-quadratic schedule,
# which is appropriate for the dev checkpoint but not the distilled artifact.
LTX_DISTILLED_TIMESTEPS = [1000, 900, 700, 500, 300, 200, 100, 40]
FRAMEPACK_BASE_REPO = "hunyuanvideo-community/HunyuanVideo"
FRAMEPACK_VISION_REPO = "lllyasviel/flux_redux_bfl"
WAN_VACE_MAX_SEQUENCE_LENGTH = 512
WAN_VACE_MAX_SEED = 4294967295
WAN_VACE_MAX_REFERENCE_IMAGES = 8
WAN_VACE_MAX_REFERENCE_PIXELS = 16 * 1024 * 1024


def _value_or_default(mapping: dict[str, Any], key: str, default: Any):
    value = mapping.get(key)
    return default if value is None else value


@dataclass(frozen=True)
class VideoPipelineAdapter:
    id: str
    pipeline_class: str
    diffusers_class: str
    default_repo: str
    modes: tuple[str, ...]
    output_media: tuple[str, ...] = ("video",)
    supports_mask: bool = False
    max_prompt_tokens: int | None = None
    default_audio_sample_rate: int | None = None


@dataclass(frozen=True)
class VideoModeMediaContract:
    video: str
    mask: str
    reference_images: str


_VIDEO_DYNAMIC_FIELDS = (
    "video",
    "mask",
    "reference_images",
    "conditioning_scale",
    "strength",
    "denoise_strength",
    "frame_rate",
    "last_image",
    "framepack_sampling",
    "latent_window_size",
    "true_cfg_scale",
    "secondary_guidance_scale",
    "scheduler_flow_shift",
    "guidance_scale_2",
    "use_guidance_scale_2",
    "pose_video",
    "face_video",
    "background_video",
    "segment_frame_length",
    "previous_conditioning_frames",
    "motion_encode_batch_size",
    "temporal_tile_size",
    "temporal_overlap",
    "temporal_overlap_condition_strength",
    "adain_factor",
    "prompt_segments_json",
)
_VIDEO_INPUT_FIELDS = frozenset(
    {
        "video",
        "mask",
        "reference_images",
        "last_image",
        "pose_video",
        "face_video",
        "background_video",
    }
)


def _studio_identity_binding(form_field: str) -> dict[str, Any]:
    groups = {
        "strength": "video-strength",
        "conditioningScale": "video-conditioning-scale",
    }
    if form_field not in groups:
        raise ValueError("Video field bindings must target a reviewed Studio form field.")
    return {
        "schemaVersion": 1,
        "group": groups[form_field],
        "formFields": [form_field],
        "transform": "identity",
    }


@dataclass(frozen=True)
class VideoModeFieldContract:
    visible_fields: tuple[str, ...] = ()
    required_fields: tuple[str, ...] = ()
    strength_form_field: str = "strength"

    def __post_init__(self) -> None:
        visible = set(self.visible_fields)
        required = set(self.required_fields)
        allowed = set(_VIDEO_DYNAMIC_FIELDS)
        if len(visible) != len(self.visible_fields) or not visible.issubset(allowed):
            raise ValueError("Video mode contracts must declare unique reviewed visibility fields.")
        if len(required) != len(self.required_fields) or not required.issubset(visible & _VIDEO_INPUT_FIELDS):
            raise ValueError("Video mode contracts may require only visible reviewed input fields.")
        if self.strength_form_field not in {"strength", "conditioningScale"}:
            raise ValueError("Video strength bindings must target a reviewed Studio form field.")

    def field_param_overlay(self) -> dict[str, dict[str, Any]]:
        visible = set(self.visible_fields)
        required = set(self.required_fields)
        overlay = {field: {"hidden": field not in visible} for field in _VIDEO_DYNAMIC_FIELDS}
        for field in _VIDEO_INPUT_FIELDS:
            overlay[field]["required"] = field in required
        overlay["strength"]["fieldOptions"] = {"studioBinding": _studio_identity_binding(self.strength_form_field)}
        return overlay


WAN_VACE_MODE_MEDIA_CONTRACTS = {
    "text_to_video": VideoModeMediaContract("forbidden", "forbidden", "forbidden"),
    "video_to_video": VideoModeMediaContract("required", "forbidden", "optional"),
    "video_inpaint": VideoModeMediaContract("required", "required", "optional"),
    "video_outpaint": VideoModeMediaContract("required", "required", "optional"),
    "reference_to_video": VideoModeMediaContract("forbidden", "forbidden", "required"),
    "control_to_video": VideoModeMediaContract("required", "forbidden", "optional"),
    "video_color_edit": VideoModeMediaContract("required", "forbidden", "optional"),
}


VIDEO_PIPELINE_ADAPTERS = {
    "WanVACEPipeline": VideoPipelineAdapter(
        id="wan-vace",
        pipeline_class="WanVACEPipeline",
        diffusers_class="WanVACEPipeline",
        default_repo="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        modes=(
            "text_to_video",
            "video_to_video",
            "video_inpaint",
            "video_outpaint",
            "reference_to_video",
            "control_to_video",
            "video_color_edit",
        ),
        supports_mask=True,
    ),
    "WanVideoToVideoPipeline": VideoPipelineAdapter(
        id="wan-video-to-video",
        pipeline_class="WanVideoToVideoPipeline",
        diffusers_class="WanVideoToVideoPipeline",
        default_repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        modes=("video_to_video", "video_color_edit"),
    ),
    "WanPipeline": VideoPipelineAdapter(
        id="wan-text-to-video",
        pipeline_class="WanPipeline",
        diffusers_class="WanPipeline",
        default_repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        modes=("text_to_video",),
    ),
    "Wan22Pipeline": VideoPipelineAdapter(
        id="wan-2.2-text-to-video",
        pipeline_class="Wan22Pipeline",
        diffusers_class="WanPipeline",
        default_repo="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        modes=("text_to_video",),
    ),
    "WanTI2VPipeline": VideoPipelineAdapter(
        id="wan-2.2-ti2v-5b",
        pipeline_class="WanTI2VPipeline",
        diffusers_class="WanPipeline",
        default_repo="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        modes=("text_to_video",),
        max_prompt_tokens=512,
    ),
    "WanImageToVideoPipeline": VideoPipelineAdapter(
        id="wan-image-to-video",
        pipeline_class="WanImageToVideoPipeline",
        diffusers_class="WanImageToVideoPipeline",
        default_repo="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        modes=("image_to_video",),
        max_prompt_tokens=512,
    ),
    "WanAnimatePipeline": VideoPipelineAdapter(
        id="wan-animate",
        pipeline_class="WanAnimatePipeline",
        diffusers_class="WanAnimatePipeline",
        default_repo="Wan-AI/Wan2.2-Animate-14B-Diffusers",
        modes=("character_animate", "character_replace"),
    ),
    "LTXConditionPipeline": VideoPipelineAdapter(
        id="ltx-video",
        pipeline_class="LTXConditionPipeline",
        diffusers_class="LTXConditionPipeline",
        default_repo="Lightricks/LTX-Video-0.9.8-13B-distilled",
        modes=("text_to_video", "image_to_video", "video_to_video", "reference_to_video"),
        max_prompt_tokens=128,
    ),
    "LTXI2VLongMultiPromptPipeline": VideoPipelineAdapter(
        id="ltx-video-long-i2v",
        pipeline_class="LTXI2VLongMultiPromptPipeline",
        diffusers_class="LTXI2VLongMultiPromptPipeline",
        default_repo="Lightricks/LTX-Video-0.9.8-13B-distilled",
        modes=("image_to_video",),
        max_prompt_tokens=128,
    ),
    "LTX2ConditionPipeline": VideoPipelineAdapter(
        id="ltx-2",
        pipeline_class="LTX2ConditionPipeline",
        diffusers_class="LTX2ConditionPipeline",
        default_repo="Lightricks/LTX-2",
        modes=("text_to_video", "image_to_video", "video_to_video", "reference_to_video"),
        output_media=("video", "audio"),
        max_prompt_tokens=1024,
        default_audio_sample_rate=24000,
    ),
    "HunyuanVideoFramepackPipeline": VideoPipelineAdapter(
        id="framepack",
        pipeline_class="HunyuanVideoFramepackPipeline",
        diffusers_class="HunyuanVideoFramepackPipeline",
        default_repo="lllyasviel/FramePackI2V_HY",
        modes=("image_to_video",),
        max_prompt_tokens=256,
    ),
}


def _video_field_contract(
    *visible_fields: str,
    required_fields: tuple[str, ...] = (),
    strength_form_field: str = "strength",
) -> VideoModeFieldContract:
    return VideoModeFieldContract(
        visible_fields=visible_fields,
        required_fields=required_fields,
        strength_form_field=strength_form_field,
    )


_VACE_GUIDANCE_FIELDS = ("conditioning_scale", "guidance_scale_2", "use_guidance_scale_2")
_ANIMATE_FIELDS = (
    "reference_images",
    "pose_video",
    "face_video",
    "segment_frame_length",
    "previous_conditioning_frames",
    "motion_encode_batch_size",
)
_LTX_LONG_FIELDS = (
    "reference_images",
    "strength",
    "frame_rate",
    "temporal_tile_size",
    "temporal_overlap",
    "temporal_overlap_condition_strength",
    "adain_factor",
    "prompt_segments_json",
)
VIDEO_MODE_FIELD_CONTRACTS = {
    "WanVACEPipeline": {
        "text_to_video": _video_field_contract(*_VACE_GUIDANCE_FIELDS),
        "video_to_video": _video_field_contract(
            "video", "reference_images", *_VACE_GUIDANCE_FIELDS, required_fields=("video",)
        ),
        "video_inpaint": _video_field_contract(
            "video",
            "mask",
            "reference_images",
            *_VACE_GUIDANCE_FIELDS,
            required_fields=("video", "mask"),
        ),
        "video_outpaint": _video_field_contract(
            "video",
            "mask",
            "reference_images",
            *_VACE_GUIDANCE_FIELDS,
            required_fields=("video", "mask"),
        ),
        "reference_to_video": _video_field_contract(
            "reference_images",
            *_VACE_GUIDANCE_FIELDS,
            required_fields=("reference_images",),
        ),
        "control_to_video": _video_field_contract(
            "video", "reference_images", *_VACE_GUIDANCE_FIELDS, required_fields=("video",)
        ),
        "video_color_edit": _video_field_contract(
            "video", "reference_images", *_VACE_GUIDANCE_FIELDS, required_fields=("video",)
        ),
    },
    "WanVideoToVideoPipeline": {
        mode: _video_field_contract("video", "strength", required_fields=("video",))
        for mode in ("video_to_video", "video_color_edit")
    },
    "WanPipeline": {"text_to_video": _video_field_contract("scheduler_flow_shift")},
    "Wan22Pipeline": {"text_to_video": _video_field_contract("scheduler_flow_shift")},
    "WanTI2VPipeline": {"text_to_video": _video_field_contract("scheduler_flow_shift")},
    "WanImageToVideoPipeline": {
        "image_to_video": _video_field_contract(
            "reference_images",
            "last_image",
            "secondary_guidance_scale",
            required_fields=("reference_images",),
        )
    },
    "WanAnimatePipeline": {
        "character_animate": _video_field_contract(
            *_ANIMATE_FIELDS,
            required_fields=("reference_images", "pose_video", "face_video"),
        ),
        "character_replace": _video_field_contract(
            *_ANIMATE_FIELDS,
            "background_video",
            "mask",
            required_fields=("reference_images", "pose_video", "face_video", "background_video", "mask"),
        ),
    },
    "LTXConditionPipeline": {
        "text_to_video": _video_field_contract("frame_rate"),
        "image_to_video": _video_field_contract(
            "reference_images", "strength", "frame_rate", required_fields=("reference_images",)
        ),
        "video_to_video": _video_field_contract(
            "video",
            "strength",
            "denoise_strength",
            "frame_rate",
            required_fields=("video",),
            strength_form_field="conditioningScale",
        ),
        "reference_to_video": _video_field_contract(
            "reference_images", "strength", "frame_rate", required_fields=("reference_images",)
        ),
    },
    "LTXI2VLongMultiPromptPipeline": {
        "image_to_video": _video_field_contract(*_LTX_LONG_FIELDS, required_fields=("reference_images",))
    },
    "LTX2ConditionPipeline": {
        "text_to_video": _video_field_contract("frame_rate"),
        "image_to_video": _video_field_contract(
            "reference_images", "strength", "frame_rate", required_fields=("reference_images",)
        ),
        "video_to_video": _video_field_contract(
            "video", "strength", "frame_rate", required_fields=("video",), strength_form_field="conditioningScale"
        ),
        "reference_to_video": _video_field_contract(
            "reference_images", "strength", "frame_rate", required_fields=("reference_images",)
        ),
    },
    "HunyuanVideoFramepackPipeline": {
        "image_to_video": _video_field_contract(
            "reference_images",
            "last_image",
            "framepack_sampling",
            "latent_window_size",
            "true_cfg_scale",
            required_fields=("reference_images",),
        )
    },
}


def get_video_mode_field_contract(adapter: VideoPipelineAdapter, mode: str) -> VideoModeFieldContract:
    contracts = VIDEO_MODE_FIELD_CONTRACTS.get(adapter.pipeline_class)
    if contracts is None or tuple(contracts) != adapter.modes:
        raise RuntimeError(f"Video adapter {adapter.pipeline_class} has an incomplete reviewed field contract.")
    contract = contracts.get(mode)
    if contract is None:
        raise ValueError(f"{adapter.pipeline_class} does not support video mode {mode}.")
    return contract


VIDEO_PIPELINE_LOAD_HANDLERS = {
    "WanVACEPipeline": "_load_wan_vace",
    "WanVideoToVideoPipeline": "_load_wan_video_to_video",
    "WanPipeline": "_load_wan_text_to_video",
    "Wan22Pipeline": "_load_wan_text_to_video",
    "WanTI2VPipeline": "_load_wan_text_to_video",
    "WanImageToVideoPipeline": "_load_wan_image_to_video",
    "WanAnimatePipeline": "_load_wan_animate",
    "LTXConditionPipeline": "_load_ltx",
    "LTXI2VLongMultiPromptPipeline": "_load_ltx_long",
    "LTX2ConditionPipeline": "_load_ltx2",
    "HunyuanVideoFramepackPipeline": "_load_framepack",
}


VIDEO_PIPELINE_EXECUTE_HANDLERS = {
    "WanVACEPipeline": "_execute_wan_vace",
    "WanVideoToVideoPipeline": "_execute_wan_video_to_video",
    "WanPipeline": "_execute_wan_text_to_video",
    "Wan22Pipeline": "_execute_wan_text_to_video",
    "WanTI2VPipeline": "_execute_wan_text_to_video",
    "WanImageToVideoPipeline": "_execute_wan_image_to_video",
    "WanAnimatePipeline": "_execute_wan_animate",
    "LTXConditionPipeline": "_execute_ltx",
    "LTXI2VLongMultiPromptPipeline": "_execute_ltx_long",
    "LTX2ConditionPipeline": "_execute_ltx2",
    "HunyuanVideoFramepackPipeline": "_execute_framepack",
}


def get_video_pipeline_adapter(name: Any) -> VideoPipelineAdapter:
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError("A registered Diffusers video pipeline class is required.")
    adapter = VIDEO_PIPELINE_ADAPTERS.get(name)
    if adapter is None:
        supported = ", ".join(sorted(VIDEO_PIPELINE_ADAPTERS))
        raise ValueError(f"Unsupported Diffusers video pipeline class {name}. Supported classes: {supported}.")
    return adapter


def _canonical_video_model_source(source: Any) -> str:
    if not isinstance(source, str) or not source or source != source.strip():
        raise ValueError("Diffusers video model source must be exactly hub or local.")
    normalized = source.casefold()
    if normalized not in {"hub", "local"}:
        raise ValueError("Diffusers video model source must be exactly hub or local.")
    return normalized


def _validated_video_hub_repository(value: str) -> str:
    if value.count("/") != 1:
        raise ValueError("Diffusers video Hub models must use an exact namespace/repository ID.")
    try:
        validate_hf_repo_id(value)
    except ValueError as error:
        raise ValueError("Diffusers video Hub models must use an exact namespace/repository ID.") from error
    try:
        resolves_locally = Path(value).expanduser().exists()
    except (OSError, RuntimeError) as error:
        raise ValueError("Diffusers video Hub model identity could not be validated.") from error
    if resolves_locally:
        raise ValueError(
            "Diffusers video Hub model resolves to an existing local filesystem target. "
            "Select source=local for local models."
        )
    return value


def _validated_video_local_model_directory(value: str) -> str:
    try:
        resolved = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Local Diffusers video model directory does not exist: {value}") from error
    if not resolved.is_dir():
        raise ValueError(f"Local Diffusers video model selection must be a directory: {value}")
    return str(resolved)


def _resolve_adapter_model_selection(adapter: VideoPipelineAdapter, value: Any):
    """Replace managed adapter defaults while preserving custom selections.

    The generic loader inherits one persisted model selector. A class-only
    change can therefore leave any previous adapter's managed default behind.
    Registered Hub defaults follow the adapter selection, while an explicit
    custom Hub repository or any local selection remains untouched.
    """

    if value is None or (isinstance(value, str) and not value.strip()):
        source = "hub"
        selected = adapter.default_repo
    elif isinstance(value, dict):
        source = _canonical_video_model_source(value.get("source"))
        raw_selected = value.get("value")
        if not isinstance(raw_selected, str):
            raise ValueError("Diffusers video model value must be a repository ID or local path string.")
        selected = raw_selected.strip()
        if not selected:
            if source == "hub":
                selected = adapter.default_repo
            else:
                raise ValueError("A local Diffusers video model path is required.")
    elif isinstance(value, str):
        source = "hub"
        selected = value.strip()
    else:
        raise ValueError("Diffusers video model selection must be a repository ID or a hub/local selection object.")

    if source == "hub":
        selected = _validated_video_hub_repository(selected)
    else:
        selected = _validated_video_local_model_directory(selected)
    managed_defaults = {candidate.default_repo.casefold() for candidate in VIDEO_PIPELINE_ADAPTERS.values()}
    selected_key = selected.casefold()
    if source == "hub" and selected_key in managed_defaults:
        return {"source": "hub", "value": adapter.default_repo}
    return {"source": source, "value": selected}


def _resolve_loader_revision(model_selection: Any, model_id: str, revision: Any) -> str | None:
    source = model_selection.get("source") if isinstance(model_selection, dict) else None
    if source == "local":
        return None

    catalog_pin = catalog_revision(model_id)
    if revision is None or revision == "":
        if catalog_pin is not None:
            return catalog_pin
        raise ValueError(
            f"Custom Hugging Face video repository {model_id!r} requires an explicit immutable "
            "lowercase 40-character commit SHA revision."
        )
    if (
        not isinstance(revision, str)
        or revision != revision.strip()
        or revision != revision.lower()
        or not IMMUTABLE_HUB_REVISION.fullmatch(revision)
    ):
        raise ValueError("Hugging Face video revision must be an exact lowercase 40-character commit SHA.")
    if catalog_pin is not None and revision != catalog_pin:
        raise ValueError(
            f"Cataloged Hugging Face video repository {model_id!r} is pinned to {catalog_pin}; "
            f"the requested revision {revision} does not match."
        )
    return revision


_MISSING_VIDEO_PIPELINE_TAG = object()


def _pipeline_adapter(pipeline: Any) -> VideoPipelineAdapter:
    runtime_class = type(pipeline).__name__
    runtime_candidates = [
        adapter
        for adapter in VIDEO_PIPELINE_ADAPTERS.values()
        if runtime_class in {adapter.pipeline_class, adapter.diffusers_class}
    ]
    adapter_name = getattr(pipeline, "_modiff_video_pipeline_class", _MISSING_VIDEO_PIPELINE_TAG)
    if adapter_name is not _MISSING_VIDEO_PIPELINE_TAG:
        adapter = get_video_pipeline_adapter(adapter_name)
        if runtime_candidates and adapter not in runtime_candidates:
            runtime_names = ", ".join(sorted(candidate.pipeline_class for candidate in runtime_candidates))
            raise ValueError(
                f"Diffusers video pipeline identity is inconsistent: runtime class {runtime_class} supports "
                f"{runtime_names}, but the pipeline is tagged as {adapter.pipeline_class}."
            )
        tagged_repo = getattr(pipeline, "_modiff_video_repo", None)
        if isinstance(tagged_repo, str) and tagged_repo.strip():
            repository_key = tagged_repo.strip().casefold()
            managed_repo_adapters = [
                candidate
                for candidate in VIDEO_PIPELINE_ADAPTERS.values()
                if candidate.default_repo.casefold() == repository_key
            ]
            if managed_repo_adapters and adapter not in managed_repo_adapters:
                repository_names = ", ".join(sorted(candidate.pipeline_class for candidate in managed_repo_adapters))
                raise ValueError(
                    "Diffusers video pipeline identity is inconsistent: managed repository "
                    f"{tagged_repo.strip()!r} supports {repository_names}, but the pipeline is tagged as "
                    f"{adapter.pipeline_class}."
                )
        return adapter

    if not runtime_candidates:
        raise ValueError(
            f"Cannot recover a registered Diffusers video adapter from untagged runtime class {runtime_class}."
        )

    reviewed_repo = getattr(pipeline, "_modiff_video_repo", None)
    if not isinstance(reviewed_repo, str) or not reviewed_repo.strip():
        raise ValueError(
            f"Cannot recover untagged Diffusers video runtime class {runtime_class} without an exact reviewed "
            "repository identity. Reload it through the generic video loader."
        )
    candidates = [adapter for adapter in runtime_candidates if adapter.default_repo == reviewed_repo.strip()]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            f"Cannot recover untagged Diffusers video runtime class {runtime_class} from unreviewed repository "
            f"{reviewed_repo!r}. Reload it through the generic video loader."
        )
    candidate_names = ", ".join(sorted(adapter.pipeline_class for adapter in candidates))
    raise ValueError(
        f"Untagged Diffusers video runtime class {runtime_class} is ambiguous across adapters: {candidate_names}. "
        "Reload it through the generic video loader so its exact adapter is tagged."
    )


def _adapter_signal(adapter: VideoPipelineAdapter) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "library": "diffusers",
        "mediaKind": "video",
        "pipelineClass": adapter.pipeline_class,
        "modes": list(adapter.modes),
    }


def _media_frame_container_family(frame: Any, *, field_name: str, index: int) -> str:
    label = f"{field_name} frame {index + 1}"
    if isinstance(frame, Image.Image):
        return "pil"
    if isinstance(frame, np.ndarray):
        return "numpy"

    frame_type = type(frame)
    module_name = str(getattr(frame_type, "__module__", ""))
    is_torch_like = (
        module_name.startswith("torch")
        and callable(getattr(frame, "detach", None))
        and hasattr(frame, "device")
        and hasattr(frame, "dtype")
        and hasattr(frame, "shape")
    )
    if is_torch_like:
        return "torch"
    raise ValueError(f"Wan VACE {label} must be a PIL image, NumPy array, or Torch tensor-like image.")


def _media_frame_spatial_size(frame: Any, *, field_name: str, index: int) -> tuple[int, int]:
    """Validate one image-like frame without importing a heavyweight runtime."""

    family = _media_frame_container_family(frame, field_name=field_name, index=index)
    size = getattr(frame, "size", None)

    label = f"{field_name} frame {index + 1}"
    if family == "pil":
        try:
            width, height = (int(value) for value in size)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Wan VACE {label} has an invalid PIL spatial size.") from error
    else:
        try:
            shape = tuple(int(value) for value in frame.shape)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(f"Wan VACE {label} has an invalid array/tensor shape.") from error
        if len(shape) == 2:
            height, width = shape
        elif len(shape) == 3:
            # Diffusers' image/video processors use NumPy HWC and Torch CHW.
            # Accepting the opposite layout here is unsafe for masked VACE
            # modes: neutralization broadcasts along those canonical channel
            # axes before the upstream processor runs.
            if family == "numpy":
                if shape[-1] not in {1, 3, 4}:
                    raise ValueError(f"Wan VACE {label} NumPy images must use HWC layout with 1, 3, or 4 channels.")
                height, width = shape[0], shape[1]
            else:
                if shape[0] not in {1, 3, 4}:
                    raise ValueError(
                        f"Wan VACE {label} Torch tensor-like images must use CHW layout with 1, 3, or 4 channels."
                    )
                height, width = shape[1], shape[2]
        else:
            raise ValueError(f"Wan VACE {label} must be a 2D or 3D image frame; received shape {shape}.")

    if width <= 0 or height <= 0:
        raise ValueError(f"Wan VACE {label} must have positive spatial dimensions; received {width}x{height}.")
    return width, height


def _validate_media_sequence(
    frames: list[Any] | None,
    *,
    field_name: str,
    uniform_spatial_size: bool,
) -> list[tuple[int, int]]:
    if frames is None:
        return []
    families = [
        _media_frame_container_family(frame, field_name=field_name, index=index) for index, frame in enumerate(frames)
    ]
    if any(family != families[0] for family in families[1:]):
        received = ", ".join(dict.fromkeys(families))
        raise ValueError(f"Wan VACE {field_name} frames must use one container family; received {received}.")
    sizes = [
        _media_frame_spatial_size(frame, field_name=field_name, index=index) for index, frame in enumerate(frames)
    ]
    if uniform_spatial_size and any(size != sizes[0] for size in sizes[1:]):
        raise ValueError(f"Wan VACE {field_name} frames must all have the same spatial dimensions.")
    return sizes


def _normalize_wan_vace_reference_images(value: Any) -> list[Image.Image] | None:
    references = ensure_reference_images(value)
    if references is None:
        return None

    nested = [item for item in references if isinstance(item, (list, tuple))]
    if nested:
        if len(references) != 1:
            raise ValueError("Wan VACE reference images support one flat list or one nested batch only.")
        references = list(nested[0])
        if not references:
            return None
    if not all(isinstance(reference, Image.Image) for reference in references):
        raise ValueError("Wan VACE reference images must be actual PIL images.")
    if len(references) > WAN_VACE_MAX_REFERENCE_IMAGES:
        raise ValueError(f"Wan VACE accepts at most {WAN_VACE_MAX_REFERENCE_IMAGES} reference images per video.")
    return references


def _bounded_wan_vace_num_frames(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Wan VACE num_frames must be a finite integer from 1 through 241.")
    try:
        number = float(81 if value is None else value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Wan VACE num_frames must be a finite integer from 1 through 241.") from error
    if not isfinite(number) or not number.is_integer() or number < 1 or number > 241:
        raise ValueError("Wan VACE num_frames must be a finite integer from 1 through 241.")
    return int(number)


def _bounded_wan_vace_int(
    value: Any,
    *,
    default: int,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Wan VACE {label} must be a finite integer from {minimum} through {maximum}.")
    try:
        number = float(default if value is None else value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Wan VACE {label} must be a finite integer from {minimum} through {maximum}.") from error
    if not isfinite(number) or not number.is_integer() or number < minimum or number > maximum:
        raise ValueError(f"Wan VACE {label} must be a finite integer from {minimum} through {maximum}.")
    return int(number)


def _bounded_wan_vace_float(
    value: Any,
    *,
    default: float,
    label: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Wan VACE {label} must be finite and from {minimum:g} through {maximum:g}.")
    try:
        number = float(default if value is None else value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Wan VACE {label} must be finite and from {minimum:g} through {maximum:g}.") from error
    if not isfinite(number) or number < minimum or number > maximum:
        raise ValueError(f"Wan VACE {label} must be finite and from {minimum:g} through {maximum:g}.")
    return number


def _normalize_wan_vace_scalar_contract(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validate graph-controlled VACE resource values before Torch/upstream."""

    pipeline = kwargs.get("pipeline")
    width = _bounded_wan_vace_int(
        kwargs.get("width"),
        default=832,
        label="width",
        minimum=16,
        maximum=2048,
    )
    height = _bounded_wan_vace_int(
        kwargs.get("height"),
        default=480,
        label="height",
        minimum=16,
        maximum=2048,
    )
    validate_dimensions(width, height, pipeline)

    requested_num_frames = _bounded_wan_vace_num_frames(kwargs.get("num_frames"))
    normalized_num_frames = normalize_num_frames(requested_num_frames, pipeline)
    if normalized_num_frames > 241:
        raise ValueError(
            "Wan VACE num_frames normalization exceeds the supported maximum of 241; "
            f"received {requested_num_frames}, normalized to {normalized_num_frames}."
        )

    output_type = kwargs.get("output_type")
    output_type = "pil" if output_type is None else output_type
    if not isinstance(output_type, str) or output_type not in {"pil", "np", "pt"}:
        raise ValueError("Wan VACE output_type must be exactly one of: pil, np, pt.")

    return {
        "width": width,
        "height": height,
        "num_frames": normalized_num_frames,
        "num_inference_steps": _bounded_wan_vace_int(
            kwargs.get("num_inference_steps"),
            default=30,
            label="inference steps",
            minimum=1,
            maximum=100,
        ),
        "guidance_scale": _bounded_wan_vace_float(
            kwargs.get("guidance_scale"),
            default=5.0,
            label="guidance scale",
            minimum=0,
            maximum=20,
        ),
        "guidance_scale_2": _bounded_wan_vace_float(
            kwargs.get("guidance_scale_2"),
            default=0.0,
            label="secondary guidance scale",
            minimum=0,
            maximum=20,
        ),
        "conditioning_scale": _bounded_wan_vace_float(
            kwargs.get("conditioning_scale"),
            default=1.0,
            label="conditioning scale",
            minimum=0,
            maximum=2,
        ),
        "seed": _bounded_wan_vace_int(
            kwargs.get("seed"),
            default=0,
            label="seed",
            minimum=0,
            maximum=WAN_VACE_MAX_SEED,
        ),
        "num_videos_per_prompt": _bounded_wan_vace_int(
            kwargs.get("num_videos_per_prompt"),
            default=1,
            label="videos per prompt",
            minimum=1,
            maximum=1,
        ),
        "output_type": output_type,
        "max_sequence_length": _bounded_wan_vace_int(
            kwargs.get("max_sequence_length"),
            default=WAN_VACE_MAX_SEQUENCE_LENGTH,
            label="max sequence length",
            minimum=1,
            maximum=WAN_VACE_MAX_SEQUENCE_LENGTH,
        ),
    }


def _validate_wan_vace_media_contract(mode: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    contract = WAN_VACE_MODE_MEDIA_CONTRACTS.get(mode)
    if contract is None:
        raise ValueError(f"WanVACEPipeline does not have a media contract for video mode {mode}.")

    media = {
        "video": ensure_video_list(kwargs.get("video"), "video"),
        "mask": ensure_video_list(kwargs.get("mask"), "mask"),
        "reference_images": _normalize_wan_vace_reference_images(kwargs.get("reference_images")),
    }
    labels = {
        "video": "source/control video",
        "mask": "mask video",
        "reference_images": "reference images",
    }
    for field in ("video", "mask", "reference_images"):
        requirement = getattr(contract, field)
        if requirement not in {"required", "optional", "forbidden"}:
            raise RuntimeError(f"Wan VACE {mode} has an invalid {field} media requirement {requirement!r}.")
        present = media[field] is not None
        if requirement == "required" and not present:
            raise ValueError(f"Wan VACE {mode} requires {labels[field]}.")
        if requirement == "forbidden" and present:
            raise ValueError(f"Wan VACE {mode} does not accept {labels[field]}.")

    video_sizes = _validate_media_sequence(
        media["video"], field_name="source/control video", uniform_spatial_size=True
    )
    mask_sizes = _validate_media_sequence(media["mask"], field_name="mask video", uniform_spatial_size=True)
    reference_sizes = _validate_media_sequence(
        media["reference_images"],
        field_name="reference image",
        uniform_spatial_size=False,
    )
    reference_pixels = sum(width * height for width, height in reference_sizes)
    if reference_pixels > WAN_VACE_MAX_REFERENCE_PIXELS:
        raise ValueError(
            f"Wan VACE reference images exceed the {WAN_VACE_MAX_REFERENCE_PIXELS}-pixel cumulative input limit."
        )

    if media["video"] is not None and media["mask"] is not None:
        if len(media["video"]) != len(media["mask"]):
            raise ValueError(
                f"Wan VACE video/mask frame count mismatch: {len(media['video'])} vs {len(media['mask'])}."
            )
        if any(video_size != mask_size for video_size, mask_size in zip(video_sizes, mask_sizes)):
            raise ValueError("Wan VACE source/control video and mask frames must have matching spatial dimensions.")
        for index, (video_frame, mask_frame) in enumerate(zip(media["video"], media["mask"])):
            video_family = _media_frame_container_family(
                video_frame,
                field_name="source/control video",
                index=index,
            )
            mask_family = _media_frame_container_family(mask_frame, field_name="mask video", index=index)
            if video_family != mask_family:
                raise ValueError(
                    "Wan VACE source/control video and mask frame "
                    f"{index + 1} must use the same container family; received {video_family} and {mask_family}."
                )
            if video_family in {"numpy", "torch"}:
                video_ndim = len(tuple(video_frame.shape))
                mask_ndim = len(tuple(mask_frame.shape))
                if video_ndim == 2 and mask_ndim != 2:
                    raise ValueError(
                        "Wan VACE 2D source/control video frames require 2D mask frames; "
                        f"frame {index + 1} received a {mask_ndim}D mask."
                    )

    scalar_values = _normalize_wan_vace_scalar_contract(kwargs)
    normalized_num_frames = scalar_values["num_frames"]
    if media["video"] is not None and len(media["video"]) != normalized_num_frames:
        raise ValueError(
            f"Wan VACE {mode} received {len(media['video'])} conditioned video frames, but normalized "
            f"num_frames is {normalized_num_frames}."
        )
    if (
        media["video"] is not None
        and normalized_num_frames > WAN_VACE_NATIVE_CHUNK_FRAMES
        and scalar_values["output_type"] != "pil"
    ):
        raise ValueError(
            "Segmented Wan VACE conditioned video currently requires output_type=pil; "
            "NumPy and Torch chunk concatenation are not supported."
        )

    return {**media, **scalar_values}


DEFAULT_VIDEO_CONTRACT = _adapter_signal(VIDEO_PIPELINE_ADAPTERS["WanVACEPipeline"])


def _require_video_mode(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("An exact non-empty Diffusers video mode is required.")
    return value


def _normalize_ltx_frames(value: int) -> int:
    if value < 1:
        return 1
    remainder = (value - 1) % 8
    return value if remainder == 0 else value + (8 - remainder)


def _validate_ltx_dimensions(width: int, height: int):
    if width % 32 or height % 32:
        raise ValueError(f"LTX Video width and height must be divisible by 32; received {width}x{height}.")


def _validate_prompt_token_limit(pipeline: Any, prompt: str | None, label: str, limit: int | None):
    if not prompt or not limit:
        return
    tokenizer = getattr(pipeline, "tokenizer", None)
    if not callable(tokenizer):
        return
    encoded = tokenizer(prompt, add_special_tokens=True, truncation=False)
    token_ids = encoded.get("input_ids") if hasattr(encoded, "get") else None
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    token_count = len(token_ids or [])
    if token_count > limit:
        raise ValueError(
            f"LTX {label} uses {token_count} tokens, but this artifact supports at most {limit}. "
            "Shorten the text so motion and preservation constraints are not truncated."
        )


def _install_ltx_dynamic_shift(pipeline: Any, width: int, height: int, num_frames: int):
    """Inject the official resolution-dependent scheduler shift for LTX Condition.

    Diffusers' LTXPipeline calculates and passes ``mu`` itself. Some released
    LTXConditionPipeline versions use the same dynamic FlowMatch scheduler but
    omit that argument. Wrap the scheduler call for this invocation so both
    pipeline variants use the model's official sequence-length shift without
    disabling dynamic scheduling.
    """

    scheduler = getattr(pipeline, "scheduler", None)
    config = getattr(scheduler, "config", None)
    if scheduler is None or not config or not bool(config.get("use_dynamic_shifting", False)):
        return None
    original = getattr(scheduler, "set_timesteps", None)
    if not callable(original):
        return None

    temporal_ratio = int(getattr(pipeline, "vae_temporal_compression_ratio", 8))
    spatial_ratio = int(getattr(pipeline, "vae_spatial_compression_ratio", 32))
    latent_frames = (num_frames - 1) // temporal_ratio + 1
    sequence_length = latent_frames * (height // spatial_ratio) * (width // spatial_ratio)
    base_sequence_length = int(config.get("base_image_seq_len", 256))
    max_sequence_length = int(config.get("max_image_seq_len", 4096))
    base_shift = float(config.get("base_shift", 0.5))
    max_shift = float(config.get("max_shift", 1.15))
    slope = (max_shift - base_shift) / (max_sequence_length - base_sequence_length)
    mu = sequence_length * slope + (base_shift - slope * base_sequence_length)

    @wraps(original)
    def set_timesteps_with_mu(*args, **kwargs):
        kwargs.setdefault("mu", mu)
        return original(*args, **kwargs)

    scheduler.set_timesteps = set_timesteps_with_mu
    return scheduler, original


class LoadPipeline(WanVACELoadPipeline):
    """Load a registered Diffusers video pipeline through a stable facade."""

    label = "Load Diffusers Video Pipeline"
    category = "Diffusers Video"
    params = {
        **WanVACELoadPipeline.params,
        "model_id": {
            **WanVACELoadPipeline.params["model_id"],
            "onChange": "select_adapter",
        },
        "pipeline": {
            "label": "Pipeline",
            "display": "output",
            "type": "video_diffusion_pipeline",
            "signal": {
                "direction": "output",
                "origin": "pipeline_class",
                "value": DEFAULT_VIDEO_CONTRACT,
            },
        },
        "pipeline_class": {
            "label": "Pipeline Class",
            "type": "string",
            "options": list(VIDEO_PIPELINE_ADAPTERS),
            "default": "WanVACEPipeline",
            "fieldOptions": {"noValidation": True},
            "onChange": "select_adapter",
        },
        "resolved_artifact": {"label": "Resolved Artifact", "display": "output", "type": "string"},
    }

    def __call__(self, **kwargs):
        # A missing, null, or malformed class is not distinguishable from a
        # stale imported graph. New graphs persist the declared default
        # explicitly, so execution can require exact identity here.
        adapter = get_video_pipeline_adapter(kwargs.get("pipeline_class"))
        values = dict(kwargs)
        values["model_id"] = _resolve_adapter_model_selection(adapter, values.get("model_id"))
        model_id = repo_value(values["model_id"])
        values["revision"] = _resolve_loader_revision(values["model_id"], model_id, values.get("revision"))
        return super().__call__(**values)

    def execute(self, **kwargs):
        adapter = get_video_pipeline_adapter(kwargs.get("pipeline_class"))
        values = dict(kwargs)
        values["model_id"] = _resolve_adapter_model_selection(adapter, values.get("model_id"))
        model_id = repo_value(values["model_id"])
        values["revision"] = _resolve_loader_revision(values["model_id"], model_id, values.get("revision"))
        handler_name = VIDEO_PIPELINE_LOAD_HANDLERS.get(adapter.pipeline_class)
        handler = getattr(self, handler_name, None) if handler_name else None
        if not callable(handler):
            raise RuntimeError(f"No loader handler is registered for video adapter {adapter.pipeline_class}.")
        pipeline = handler(adapter, values)
        setattr(pipeline, "_modiff_video_pipeline_class", adapter.pipeline_class)
        setattr(pipeline, "_modiff_video_repo", model_id or adapter.default_repo)
        setattr(pipeline, "_modiff_video_revision", values["revision"])
        return {
            "pipeline": pipeline,
            "resolved_artifact": model_id or adapter.default_repo,
        }

    def select_adapter(self, values, ref):
        values = values if isinstance(values, dict) else {}
        adapter = get_video_pipeline_adapter(values.get("pipeline_class"))
        selected = values.get("model_id")
        resolved = _resolve_adapter_model_selection(adapter, selected)
        signal_origin = ref.get("key") if isinstance(ref, dict) else ref
        field_values = {}
        if resolved != selected:
            field_values["model_id"] = resolved
        if resolved["source"] == "local":
            field_values["revision"] = ""
        else:
            managed_revision = catalog_revision(resolved["value"])
            if managed_revision is not None:
                field_values["revision"] = managed_revision
            elif signal_origin == "model_id":
                # A custom repository selected by this action must not inherit
                # the commit belonging to the previously visible repository.
                field_values["revision"] = ""
            elif values.get("revision") not in (None, ""):
                _resolve_loader_revision(resolved, resolved["value"], values.get("revision"))
        if field_values:
            self.set_field_value(field_values)
        self.set_field_params(
            "pipeline",
            {
                "signal": {
                    "direction": "output",
                    "origin": str(signal_origin or "pipeline_class"),
                    "value": _adapter_signal(adapter),
                }
            },
        )

    def _load_wan_vace(self, adapter: VideoPipelineAdapter, kwargs: dict[str, Any]):
        result = super().execute(**kwargs)
        return result["pipeline"]

    def _load_ltx(self, adapter: VideoPipelineAdapter, kwargs: dict[str, Any]):
        from diffusers import LTXConditionPipeline
        from modules.DiffusersRuntime.main import apply_execution_recipe_to_pipeline, loader_runtime_options

        model_selection = kwargs.get("model_id")
        model_id = repo_value(model_selection) or adapter.default_repo
        dtype = str_to_dtype(kwargs.get("dtype", "bfloat16"))
        revision = _resolve_loader_revision(model_selection, model_id, kwargs.get("revision"))
        recipe, device, offload_mode, recipe_load_kwargs = loader_runtime_options(
            kwargs,
            default_device=DEFAULT_DEVICE,
            default_offload_mode=OFFLOAD_MODE_MODEL_CPU,
            direct_device_load=True,
        )
        load_kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": bool(kwargs.get("low_cpu_mem_usage", True)),
            "revision": revision,
            "local_files_only": local_files_only(model_id),
            **recipe_load_kwargs,
        }
        if CONFIG.hf.get("cache_dir"):
            load_kwargs["cache_dir"] = CONFIG.hf["cache_dir"]

        logger.info("Loading %s pipeline: %s", adapter.diffusers_class, model_id)
        self.progress(-1, phase="loading", message=f"Loading {adapter.diffusers_class}")
        pipeline = LTXConditionPipeline.from_pretrained(model_id, **load_kwargs)
        apply_execution_recipe_to_pipeline(pipeline, recipe)
        self.progress(-1, phase="component_placement", message=f"Applying {offload_mode} offload")
        apply_pipeline_offload(pipeline, mode=offload_mode, device=device, node_id=self.node_id, scope=adapter.id)
        self.mm_add(pipeline, priority=2)
        return pipeline

    def _load_ltx_long(self, adapter: VideoPipelineAdapter, kwargs: dict[str, Any]):
        from diffusers import LTXEulerAncestralRFScheduler, LTXI2VLongMultiPromptPipeline
        from modules.DiffusersRuntime.main import apply_execution_recipe_to_pipeline, loader_runtime_options

        model_selection = kwargs.get("model_id")
        model_id = repo_value(model_selection) or adapter.default_repo
        dtype = str_to_dtype(kwargs.get("dtype", "bfloat16"))
        recipe, device, offload_mode, recipe_load_kwargs = loader_runtime_options(
            kwargs,
            default_device=DEFAULT_DEVICE,
            default_offload_mode=OFFLOAD_MODE_MODEL_CPU,
            direct_device_load=True,
        )
        load_kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": bool(kwargs.get("low_cpu_mem_usage", True)),
            "revision": _resolve_loader_revision(model_selection, model_id, kwargs.get("revision")),
            "local_files_only": local_files_only(model_id),
            **recipe_load_kwargs,
        }
        if CONFIG.hf.get("cache_dir"):
            load_kwargs["cache_dir"] = CONFIG.hf["cache_dir"]
        self.progress(-1, phase="loading", message="Loading LTX long image-to-video pipeline")
        pipeline = LTXI2VLongMultiPromptPipeline.from_pretrained(model_id, **load_kwargs)
        pipeline.scheduler = LTXEulerAncestralRFScheduler.from_config(pipeline.scheduler.config)
        apply_execution_recipe_to_pipeline(pipeline, recipe)
        apply_pipeline_offload(pipeline, mode=offload_mode, device=device, node_id=self.node_id, scope=adapter.id)
        self.mm_add(pipeline, priority=2)
        return pipeline

    def _load_ltx2(self, adapter: VideoPipelineAdapter, kwargs: dict[str, Any]):
        from diffusers import LTX2ConditionPipeline
        from modules.DiffusersRuntime.main import apply_execution_recipe_to_pipeline, loader_runtime_options

        model_selection = kwargs.get("model_id")
        model_id = repo_value(model_selection) or adapter.default_repo
        dtype = str_to_dtype(kwargs.get("dtype", "bfloat16"))
        recipe, device, offload_mode, recipe_load_kwargs = loader_runtime_options(
            kwargs,
            default_device=DEFAULT_DEVICE,
            default_offload_mode="sequential_cpu",
            direct_device_load=True,
        )
        load_kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": bool(kwargs.get("low_cpu_mem_usage", True)),
            "revision": _resolve_loader_revision(model_selection, model_id, kwargs.get("revision")),
            "local_files_only": local_files_only(model_id),
            **recipe_load_kwargs,
        }
        if CONFIG.hf.get("cache_dir"):
            load_kwargs["cache_dir"] = CONFIG.hf["cache_dir"]
        self.progress(-1, phase="loading", message="Loading LTX-2 video and audio pipeline")
        pipeline = LTX2ConditionPipeline.from_pretrained(model_id, **load_kwargs)
        apply_execution_recipe_to_pipeline(pipeline, recipe)
        apply_pipeline_offload(pipeline, mode=offload_mode, device=device, node_id=self.node_id, scope=adapter.id)
        self.mm_add(pipeline, priority=2)
        return pipeline

    def _load_wan_animate(self, adapter: VideoPipelineAdapter, kwargs: dict[str, Any]):
        import torch
        from diffusers import AutoencoderKLWan, WanAnimatePipeline
        from modules.DiffusersRuntime.main import apply_execution_recipe_to_pipeline, loader_runtime_options

        model_selection = kwargs.get("model_id")
        model_id = repo_value(model_selection) or adapter.default_repo
        dtype = str_to_dtype(kwargs.get("dtype", "bfloat16"))
        recipe, device, offload_mode, recipe_load_kwargs = loader_runtime_options(
            kwargs,
            default_device=DEFAULT_DEVICE,
            default_offload_mode="sequential_cpu",
        )
        common = {
            "revision": _resolve_loader_revision(model_selection, model_id, kwargs.get("revision")),
            "low_cpu_mem_usage": bool(kwargs.get("low_cpu_mem_usage", True)),
            "local_files_only": local_files_only(model_id),
        }
        if CONFIG.hf.get("cache_dir"):
            common["cache_dir"] = CONFIG.hf["cache_dir"]
        self.progress(-1, phase="loading", message="Loading Wan Animate pipeline")
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, **common)
        pipeline = WanAnimatePipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=dtype,
            **common,
            **recipe_load_kwargs,
        )
        apply_execution_recipe_to_pipeline(pipeline, recipe)
        apply_pipeline_offload(pipeline, mode=offload_mode, device=device, node_id=self.node_id, scope=adapter.id)
        self.mm_add(pipeline, priority=2)
        return pipeline

    def _load_framepack(self, adapter: VideoPipelineAdapter, kwargs: dict[str, Any]):
        from diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
        from modules.DiffusersRuntime.main import apply_execution_recipe_to_pipeline, loader_runtime_options
        from transformers import SiglipImageProcessor, SiglipVisionModel

        model_selection = kwargs.get("model_id")
        model_id = repo_value(model_selection) or adapter.default_repo
        dtype = str_to_dtype(kwargs.get("dtype", "bfloat16"))
        revision = _resolve_loader_revision(model_selection, model_id, kwargs.get("revision"))
        recipe, device, offload_mode, recipe_load_kwargs = loader_runtime_options(
            kwargs,
            default_device=DEFAULT_DEVICE,
            default_offload_mode="sequential_cpu",
        )
        common_kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": bool(kwargs.get("low_cpu_mem_usage", True)),
        }
        if CONFIG.hf.get("cache_dir"):
            common_kwargs["cache_dir"] = CONFIG.hf["cache_dir"]

        # The FramePack repository contains only the packed transformer.  The
        # official Diffusers pipeline composes it with HunyuanVideo's base
        # components and the FLUX Redux SigLIP processor/encoder.
        transformer_kwargs = {
            **common_kwargs,
            "revision": revision,
            "local_files_only": local_files_only(model_id),
        }
        pipeline_quantization = recipe_load_kwargs.get("quantization_config")
        quant_mapping = getattr(pipeline_quantization, "quant_mapping", None)
        if isinstance(quant_mapping, dict) and quant_mapping.get("transformer") is not None:
            transformer_kwargs["quantization_config"] = quant_mapping["transformer"]

        logger.info("Loading %s transformer: %s", adapter.diffusers_class, model_id)
        self.progress(-1, phase="loading", message=f"Loading {adapter.diffusers_class}")
        transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(model_id, **transformer_kwargs)
        feature_extractor = SiglipImageProcessor.from_pretrained(
            FRAMEPACK_VISION_REPO,
            subfolder="feature_extractor",
            revision=require_catalog_revision(FRAMEPACK_VISION_REPO),
            local_files_only=local_files_only(FRAMEPACK_VISION_REPO),
            **({"cache_dir": common_kwargs["cache_dir"]} if "cache_dir" in common_kwargs else {}),
        )
        image_encoder = SiglipVisionModel.from_pretrained(
            FRAMEPACK_VISION_REPO,
            subfolder="image_encoder",
            revision=require_catalog_revision(FRAMEPACK_VISION_REPO),
            torch_dtype=dtype,
            low_cpu_mem_usage=common_kwargs["low_cpu_mem_usage"],
            local_files_only=local_files_only(FRAMEPACK_VISION_REPO),
            **({"cache_dir": common_kwargs["cache_dir"]} if "cache_dir" in common_kwargs else {}),
        )
        pipeline = HunyuanVideoFramepackPipeline.from_pretrained(
            FRAMEPACK_BASE_REPO,
            transformer=transformer,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            revision=require_catalog_revision(FRAMEPACK_BASE_REPO),
            local_files_only=local_files_only(FRAMEPACK_BASE_REPO),
            **common_kwargs,
            **recipe_load_kwargs,
        )
        apply_execution_recipe_to_pipeline(pipeline, recipe)
        self.progress(-1, phase="component_placement", message=f"Applying {offload_mode} offload")
        apply_pipeline_offload(pipeline, mode=offload_mode, device=device, node_id=self.node_id, scope=adapter.id)
        self.mm_add(pipeline, priority=2)
        return pipeline

    def _load_wan_video_to_video(self, adapter: VideoPipelineAdapter, kwargs: dict[str, Any]):
        import torch
        from diffusers import AutoencoderKLWan, WanVideoToVideoPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        from modules.DiffusersRuntime.main import apply_execution_recipe_to_pipeline, loader_runtime_options

        model_selection = kwargs.get("model_id")
        model_id = repo_value(model_selection) or adapter.default_repo
        dtype = str_to_dtype(kwargs.get("dtype", "bfloat16"))
        revision = _resolve_loader_revision(model_selection, model_id, kwargs.get("revision"))
        recipe, device, offload_mode, recipe_load_kwargs = loader_runtime_options(
            kwargs,
            default_device=DEFAULT_DEVICE,
            default_offload_mode=OFFLOAD_MODE_MODEL_CPU,
        )
        load_kwargs = {
            "low_cpu_mem_usage": bool(kwargs.get("low_cpu_mem_usage", True)),
            "revision": revision,
            "local_files_only": local_files_only(model_id),
            **recipe_load_kwargs,
        }
        if CONFIG.hf.get("cache_dir"):
            load_kwargs["cache_dir"] = CONFIG.hf["cache_dir"]

        logger.info("Loading %s pipeline: %s", adapter.diffusers_class, model_id)
        self.progress(-1, phase="loading", message=f"Loading {adapter.diffusers_class}")
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            **{key: value for key, value in load_kwargs.items() if key not in recipe_load_kwargs},
        )
        pipeline = WanVideoToVideoPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=dtype,
            **load_kwargs,
        )
        if adapter.pipeline_class != "WanTI2VPipeline":
            pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, flow_shift=3.0)
        apply_execution_recipe_to_pipeline(pipeline, recipe)
        self.progress(-1, phase="component_placement", message=f"Applying {offload_mode} offload")
        apply_pipeline_offload(pipeline, mode=offload_mode, device=device, node_id=self.node_id, scope=adapter.id)
        self.mm_add(pipeline, priority=2)
        return pipeline

    def _load_wan_image_to_video(self, adapter: VideoPipelineAdapter, kwargs: dict[str, Any]):
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        from modules.DiffusersRuntime.main import apply_execution_recipe_to_pipeline, loader_runtime_options

        model_selection = kwargs.get("model_id")
        model_id = repo_value(model_selection) or adapter.default_repo
        dtype = str_to_dtype(kwargs.get("dtype", "bfloat16"))
        revision = _resolve_loader_revision(model_selection, model_id, kwargs.get("revision"))
        recipe, device, offload_mode, recipe_load_kwargs = loader_runtime_options(
            kwargs,
            default_device=DEFAULT_DEVICE,
            default_offload_mode=OFFLOAD_MODE_MODEL_CPU,
        )
        common = {
            "low_cpu_mem_usage": bool(kwargs.get("low_cpu_mem_usage", True)),
            "revision": revision,
            "local_files_only": local_files_only(model_id),
        }
        if CONFIG.hf.get("cache_dir"):
            common["cache_dir"] = CONFIG.hf["cache_dir"]

        logger.info("Loading %s pipeline: %s", adapter.diffusers_class, model_id)
        self.progress(-1, phase="loading", message=f"Loading {adapter.diffusers_class}")
        # Wan's VAE is numerically sensitive and the official recipe keeps it
        # in FP32. The two denoisers and text encoder retain the requested load
        # dtype or per-component quantization from the execution recipe.
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            **common,
        )
        pipeline = WanImageToVideoPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=dtype,
            **common,
            **recipe_load_kwargs,
        )
        apply_execution_recipe_to_pipeline(pipeline, recipe)
        self.progress(-1, phase="component_placement", message=f"Applying {offload_mode} offload")
        apply_pipeline_offload(pipeline, mode=offload_mode, device=device, node_id=self.node_id, scope=adapter.id)
        self.mm_add(pipeline, priority=2)
        return pipeline

    def _load_wan_text_to_video(self, adapter: VideoPipelineAdapter, kwargs: dict[str, Any]):
        import torch
        from diffusers import AutoencoderKLWan, WanPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        from modules.DiffusersRuntime.main import apply_execution_recipe_to_pipeline, loader_runtime_options

        model_selection = kwargs.get("model_id")
        model_id = repo_value(model_selection) or adapter.default_repo
        dtype = str_to_dtype(kwargs.get("dtype", "bfloat16"))
        revision = _resolve_loader_revision(model_selection, model_id, kwargs.get("revision"))
        recipe, device, offload_mode, recipe_load_kwargs = loader_runtime_options(
            kwargs,
            default_device=DEFAULT_DEVICE,
            default_offload_mode=OFFLOAD_MODE_MODEL_CPU,
        )
        load_kwargs = {
            "low_cpu_mem_usage": bool(kwargs.get("low_cpu_mem_usage", True)),
            "revision": revision,
            "local_files_only": local_files_only(model_id),
            **recipe_load_kwargs,
        }
        if CONFIG.hf.get("cache_dir"):
            load_kwargs["cache_dir"] = CONFIG.hf["cache_dir"]

        logger.info("Loading %s pipeline: %s", adapter.diffusers_class, model_id)
        self.progress(-1, phase="loading", message=f"Loading {adapter.diffusers_class}")
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            **{key: value for key, value in load_kwargs.items() if key not in recipe_load_kwargs},
        )
        pipeline = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype, **load_kwargs)
        # Wan's upstream 1.3B quality recipe recommends shift 8-12 and guidance
        # 6. Use the conservative lower end for the base 1.3B text-to-video
        # adapter. Wan 2.2 checkpoints carry their own scheduler contracts and
        # must not be overwritten with a Wan 2.1 value.
        if adapter.pipeline_class == "WanPipeline":
            pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, flow_shift=8.0)
        apply_execution_recipe_to_pipeline(pipeline, recipe)
        self.progress(-1, phase="component_placement", message=f"Applying {offload_mode} offload")
        apply_pipeline_offload(pipeline, mode=offload_mode, device=device, node_id=self.node_id, scope=adapter.id)
        self.mm_add(pipeline, priority=2)
        return pipeline


class Generate(WanVACEGenerate):
    """Generate or condition video through the selected family adapter."""

    label = "Diffusers Video Generate"
    category = "Diffusers Video"
    params = {
        **WanVACEGenerate.params,
        "seed": {
            "label": "Seed",
            "type": "int",
            "display": "random",
            "default": 0,
            "min": 0,
            "max": 9007199254740991,
        },
        "pipeline": {
            "label": "Pipeline",
            "display": "input",
            "type": "video_diffusion_pipeline",
            "required": True,
            "onSignal": [
                {"action": "value", "target": "video_contract"},
                {"action": "exec", "data": "update_adapter_modes"},
            ],
        },
        "video_contract": {
            "label": "Video Contract",
            "type": "object",
            "default": DEFAULT_VIDEO_CONTRACT,
            "hidden": True,
        },
        "mode": {
            "label": "Mode",
            "type": "string",
            "options": sorted({mode for adapter in VIDEO_PIPELINE_ADAPTERS.values() for mode in adapter.modes}),
            "default": "text_to_video",
            "fieldOptions": {"noValidation": True},
            "onChange": "update_adapter_modes",
        },
        "frame_rate": {"label": "Frame rate", "type": "int", "default": 25, "min": 1, "max": 60},
        "strength": {
            "label": "Condition strength",
            "type": "float",
            "default": 1.0,
            "min": 0,
            "max": 1,
            "step": 0.05,
            "fieldOptions": {"studioBinding": _studio_identity_binding("strength")},
        },
        "denoise_strength": {
            "label": "Denoise strength",
            "type": "float",
            "default": 1.0,
            "min": 0,
            "max": 1,
            "step": 0.05,
        },
        "last_image": {"label": "Optional Last Image", "display": "input", "type": "image", "required": False},
        "framepack_sampling": {
            "label": "FramePack Sampling",
            "type": "string",
            "options": ["inverted_anti_drifting", "vanilla"],
            "default": "inverted_anti_drifting",
        },
        "latent_window_size": {"label": "FramePack Window", "type": "int", "default": 9, "min": 1, "max": 32},
        "true_cfg_scale": {"label": "True CFG", "type": "float", "default": 1.0, "min": 0, "max": 20},
        "secondary_guidance_scale": {
            "label": "Low-noise Guidance",
            "type": "float",
            "default": 3.5,
            "min": 0,
            "max": 20,
            "step": 0.1,
        },
        "scheduler_flow_shift": {
            "label": "Flow Shift",
            "type": "float",
            "default": 0,
            "min": 0,
            "max": 32,
            "step": 0.1,
        },
        "pose_video": {"label": "Pose Video", "display": "input", "type": ["video", "str"], "required": False},
        "face_video": {"label": "Face Video", "display": "input", "type": ["video", "str"], "required": False},
        "background_video": {
            "label": "Background Video",
            "display": "input",
            "type": ["video", "str"],
            "required": False,
        },
        "segment_frame_length": {"label": "Segment Frames", "type": "int", "default": 77, "min": 5, "max": 241},
        "previous_conditioning_frames": {"label": "Previous Frames", "type": "int", "default": 1, "min": 1, "max": 16},
        "motion_encode_batch_size": {"label": "Motion Batch", "type": "int", "default": 1, "min": 1, "max": 32},
        "temporal_tile_size": {"label": "Temporal Window", "type": "int", "default": 80, "min": 17, "max": 257},
        "temporal_overlap": {"label": "Temporal Overlap", "type": "int", "default": 24, "min": 1, "max": 128},
        "temporal_overlap_condition_strength": {
            "label": "Overlap Preservation",
            "type": "float",
            "default": 0.5,
            "min": 0,
            "max": 1,
            "step": 0.05,
        },
        "adain_factor": {
            "label": "Long Color Consistency",
            "type": "float",
            "default": 0.25,
            "min": 0,
            "max": 1,
            "step": 0.05,
        },
        "prompt_segments_json": {
            "label": "Timed Prompt Segments",
            "display": "textarea",
            "type": "text",
            "default": "",
        },
    }

    def __call__(self, **kwargs):
        values = dict(kwargs)
        values["mode"] = _require_video_mode(values.get("mode"))
        pipeline = values.get("pipeline")
        if pipeline is not None:
            try:
                adapter = _pipeline_adapter(pipeline)
            except ValueError:
                # Preserve the established NodeBase error wrapping for stale or
                # inconsistent pipeline identities. Execution validates it
                # authoritatively before dispatch.
                adapter = None
            if adapter is not None and adapter.pipeline_class == "WanVACEPipeline":
                values.update(_normalize_wan_vace_scalar_contract(values))
        return super().__call__(**values)

    def execute(self, **kwargs):
        _adapter, result = self._execute_with_adapter(**kwargs)
        result.pop("_audio", None)
        return result

    def update_adapter_modes(self, values, ref):
        values = values if isinstance(values, dict) else {}
        signal = values.get("video_contract")
        if not isinstance(signal, dict):
            raise ValueError("The connected video pipeline did not publish a valid adapter contract.")
        adapter = get_video_pipeline_adapter(signal.get("pipelineClass"))
        if signal != _adapter_signal(adapter):
            raise ValueError("The connected video pipeline published a stale or mismatched adapter contract.")
        current_mode = values.get("mode")
        selected_mode = current_mode if current_mode in adapter.modes else adapter.modes[0]
        field_contract = get_video_mode_field_contract(adapter, selected_mode)
        self.set_field_params(
            "mode",
            {"options": list(adapter.modes), "default": adapter.modes[0], "value": selected_mode},
        )
        for field, params in field_contract.field_param_overlay().items():
            self.set_field_params(field, params)

    def _execute_with_adapter(self, **kwargs):
        pipeline = kwargs.get("pipeline")
        if pipeline is None:
            raise ValueError("A Diffusers video pipeline is required.")
        adapter = _pipeline_adapter(pipeline)
        mode = _require_video_mode(kwargs.get("mode"))
        if mode not in adapter.modes:
            raise ValueError(f"{adapter.pipeline_class} does not support video mode {mode}.")
        values = dict(kwargs)
        values["mode"] = mode
        if adapter.pipeline_class == "WanVACEPipeline":
            values.update(_validate_wan_vace_media_contract(mode, values))
        return adapter, self._execute_adapter(pipeline, adapter, mode, values)

    def _execute_adapter(
        self,
        pipeline: Any,
        adapter: VideoPipelineAdapter,
        mode: str,
        kwargs: dict[str, Any],
    ):
        handler_name = VIDEO_PIPELINE_EXECUTE_HANDLERS.get(adapter.pipeline_class)
        handler = getattr(self, handler_name, None) if handler_name else None
        if not callable(handler):
            raise RuntimeError(f"No execution handler is registered for video adapter {adapter.pipeline_class}.")
        return handler(pipeline, adapter, mode, kwargs)

    def _execute_wan_vace(
        self,
        pipeline: Any,
        adapter: VideoPipelineAdapter,
        mode: str,
        kwargs: dict[str, Any],
    ):
        return super().execute(**kwargs)

    def _execute_wan_animate(self, pipeline: Any, adapter: VideoPipelineAdapter, mode: str, kwargs: dict[str, Any]):
        references = ensure_reference_images(kwargs.get("reference_images"))
        if not references or len(references) != 1:
            raise ValueError("Wan Animate needs exactly one character reference image.")
        pose_video = ensure_video_list(kwargs.get("pose_video"), "pose video")
        face_video = ensure_video_list(kwargs.get("face_video"), "face video")
        if not pose_video or not face_video:
            raise ValueError("Wan Animate needs preprocessed pose and face videos.")
        if len(pose_video) != len(face_video):
            raise ValueError("Wan Animate pose and face videos must contain the same number of frames.")
        call_mode = "replace" if mode == "character_replace" else "animate"
        background = ensure_video_list(kwargs.get("background_video"), "background video")
        mask = ensure_video_list(kwargs.get("mask"), "mask video")
        if call_mode == "replace" and (not background or not mask):
            raise ValueError("Wan character replacement needs background and mask videos.")
        if call_mode == "animate" and (background is not None or mask is not None):
            raise ValueError("Wan character animation does not accept background or mask videos.")
        import torch

        device = getattr(pipeline, "_execution_device", None) or "cpu"
        generator = torch.Generator(device=device).manual_seed(int(kwargs.get("seed") or 0))
        height = int(kwargs.get("height") or 720)
        width = int(kwargs.get("width") or 1280)
        self._active_pipeline = pipeline
        try:
            result = pipeline(
                image=references[0],
                pose_video=pose_video,
                face_video=face_video,
                background_video=background,
                mask_video=mask,
                prompt=ensure_single_prompt(none_if_blank(kwargs.get("prompt")), "prompt"),
                negative_prompt=ensure_single_prompt(none_if_blank(kwargs.get("negative_prompt")), "negative prompt"),
                height=height,
                width=width,
                segment_frame_length=int(kwargs.get("segment_frame_length") or 77),
                num_inference_steps=int(kwargs.get("num_inference_steps") or 20),
                mode=call_mode,
                prev_segment_conditioning_frames=int(kwargs.get("previous_conditioning_frames") or 1),
                motion_encode_batch_size=int(kwargs.get("motion_encode_batch_size") or 1),
                guidance_scale=float(_value_or_default(kwargs, "guidance_scale", 1)),
                generator=generator,
                output_type=kwargs.get("output_type") or "pil",
                return_dict=True,
                attention_kwargs=parse_json_object(kwargs.get("attention_kwargs_json"), "attention kwargs"),
                callback_on_step_end=self.pipe_callback,
                callback_on_step_end_tensor_inputs=callback_tensor_inputs(
                    kwargs.get("callback_on_step_end_tensor_inputs")
                ),
                max_sequence_length=min(int(kwargs.get("max_sequence_length") or 512), 512),
            )
        finally:
            self._active_pipeline = None
        frames = getattr(result, "frames", result)
        if isinstance(frames, list) and len(frames) == 1 and isinstance(frames[0], list):
            frames = frames[0]
        return {
            "video_out": frames,
            "width_out": width,
            "height_out": height,
            "frames_out": len(frames) if isinstance(frames, list) else len(pose_video),
        }

    def _execute_framepack(self, pipeline: Any, adapter: VideoPipelineAdapter, mode: str, kwargs: dict[str, Any]):
        if mode != "image_to_video":
            raise ValueError("FramePack supports image_to_video generation only.")
        if ensure_video_list(kwargs.get("video"), "video") is not None:
            raise ValueError("FramePack does not accept a source video; provide one opening reference image.")
        if ensure_video_list(kwargs.get("mask"), "mask") is not None:
            raise ValueError("FramePack does not accept a mask.")
        references = ensure_reference_images(kwargs.get("reference_images"))
        if not references or len(references) != 1:
            raise ValueError("FramePack needs exactly one opening reference image.")
        sampling = str(kwargs.get("framepack_sampling") or "inverted_anti_drifting")
        last_image = kwargs.get("last_image")
        if last_image is not None and sampling != "inverted_anti_drifting":
            raise ValueError("FramePack last-image guidance requires inverted_anti_drifting sampling.")
        width = int(kwargs.get("width") or 1280)
        height = int(kwargs.get("height") or 720)
        if width % 16 or height % 16:
            raise ValueError(f"FramePack width and height must be divisible by 16; received {width}x{height}.")
        num_frames = max(1, int(kwargs.get("num_frames") or 129))
        prompt = ensure_single_prompt(none_if_blank(kwargs.get("prompt")), "prompt")
        negative_prompt = ensure_single_prompt(none_if_blank(kwargs.get("negative_prompt")), "negative prompt")
        _validate_prompt_token_limit(pipeline, prompt, "prompt", adapter.max_prompt_tokens)
        import torch

        device = getattr(pipeline, "_execution_device", None) or "cpu"
        generator = torch.Generator(device=device).manual_seed(int(kwargs.get("seed") or 0))
        call_kwargs = {
            "image": references[0],
            "last_image": last_image,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "latent_window_size": int(kwargs.get("latent_window_size") or 9),
            "num_inference_steps": int(kwargs.get("num_inference_steps") or 30),
            "true_cfg_scale": float(_value_or_default(kwargs, "true_cfg_scale", 1.0)),
            "guidance_scale": float(_value_or_default(kwargs, "guidance_scale", 6.0)),
            "num_videos_per_prompt": 1,
            "generator": generator,
            "output_type": kwargs.get("output_type") or "pil",
            "return_dict": True,
            "attention_kwargs": parse_json_object(kwargs.get("attention_kwargs_json"), "attention kwargs"),
            "callback_on_step_end": self.pipe_callback,
            "callback_on_step_end_tensor_inputs": callback_tensor_inputs(
                kwargs.get("callback_on_step_end_tensor_inputs")
            ),
            "max_sequence_length": min(int(kwargs.get("max_sequence_length") or 256), 256),
            "sampling_type": sampling,
        }
        self._active_pipeline = pipeline
        try:
            result = pipeline(**call_kwargs)
        finally:
            self._active_pipeline = None
        frames = getattr(result, "frames", result)
        if isinstance(frames, list) and len(frames) == 1 and isinstance(frames[0], list):
            frames = frames[0]
        return {
            "video_out": frames,
            "width_out": width,
            "height_out": height,
            "frames_out": len(frames) if isinstance(frames, list) else num_frames,
        }

    def _execute_wan_text_to_video(
        self,
        pipeline: Any,
        adapter: VideoPipelineAdapter,
        mode: str,
        kwargs: dict[str, Any],
    ):
        if mode != "text_to_video":
            raise ValueError(f"{adapter.pipeline_class} does not support video mode {mode}.")
        if ensure_video_list(kwargs.get("video"), "video") is not None:
            raise ValueError("Wan text_to_video does not accept a source video.")
        if ensure_video_list(kwargs.get("mask"), "mask") is not None:
            raise ValueError("Wan text_to_video does not accept a mask.")
        if ensure_reference_images(kwargs.get("reference_images")):
            raise ValueError("Wan text_to_video does not accept reference images.")

        prompt = ensure_single_prompt(none_if_blank(kwargs.get("prompt")), "prompt")
        negative_prompt = ensure_single_prompt(none_if_blank(kwargs.get("negative_prompt")), "negative prompt")
        _validate_prompt_token_limit(pipeline, prompt, "prompt", adapter.max_prompt_tokens)
        _validate_prompt_token_limit(pipeline, negative_prompt, "negative prompt", adapter.max_prompt_tokens)
        import torch

        is_ti2v = adapter.pipeline_class == "WanTI2VPipeline"
        width = int(kwargs.get("width") or (1280 if is_ti2v else 832))
        height = int(kwargs.get("height") or (704 if is_ti2v else 480))
        num_frames = normalize_num_frames(int(kwargs.get("num_frames") or (121 if is_ti2v else 81)), pipeline)
        device = getattr(pipeline, "_execution_device", None) or "cpu"
        generator = torch.Generator(device=device).manual_seed(int(kwargs.get("seed", 0)))
        default_guidance = 6.0 if adapter.pipeline_class == "WanPipeline" else 5.0
        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": int(kwargs.get("num_inference_steps") or 50),
            "guidance_scale": float(kwargs.get("guidance_scale", default_guidance)),
            "num_videos_per_prompt": 1,
            "generator": generator,
            "latents": none_if_blank(kwargs.get("latents")),
            "prompt_embeds": none_if_blank(kwargs.get("prompt_embeds")),
            "negative_prompt_embeds": none_if_blank(kwargs.get("negative_prompt_embeds")),
            "output_type": kwargs.get("output_type", "pil"),
            "return_dict": True,
            "attention_kwargs": parse_json_object(kwargs.get("attention_kwargs_json"), "attention kwargs"),
            "callback_on_step_end": self.pipe_callback,
            "callback_on_step_end_tensor_inputs": callback_tensor_inputs(
                kwargs.get("callback_on_step_end_tensor_inputs")
            ),
            "max_sequence_length": int(kwargs.get("max_sequence_length", 512)),
        }
        requested_flow_shift = float(kwargs.get("scheduler_flow_shift") or 0)
        original_scheduler = None
        if requested_flow_shift > 0:
            from diffusers import UniPCMultistepScheduler

            original_scheduler = pipeline.scheduler
            pipeline.scheduler = UniPCMultistepScheduler.from_config(
                original_scheduler.config, flow_shift=requested_flow_shift
            )
        self._active_pipeline = pipeline
        try:
            result = pipeline(**call_kwargs)
        finally:
            self._active_pipeline = None
            if original_scheduler is not None:
                pipeline.scheduler = original_scheduler
        frames = getattr(result, "frames", result)
        if isinstance(frames, list) and len(frames) == 1 and isinstance(frames[0], list):
            frames = frames[0]
        return {
            "video_out": frames,
            "width_out": width,
            "height_out": height,
            "frames_out": len(frames) if isinstance(frames, list) else num_frames,
        }

    def _execute_wan_video_to_video(
        self,
        pipeline: Any,
        adapter: VideoPipelineAdapter,
        mode: str,
        kwargs: dict[str, Any],
    ):
        prompt = ensure_single_prompt(none_if_blank(kwargs.get("prompt")), "prompt")
        negative_prompt = ensure_single_prompt(none_if_blank(kwargs.get("negative_prompt")), "negative prompt")
        video = ensure_video_list(kwargs.get("video"), "video")
        if video is None:
            raise ValueError(f"Wan {mode} requires a source video.")
        if ensure_video_list(kwargs.get("mask"), "mask") is not None:
            raise ValueError(f"{adapter.pipeline_class} does not accept a mask; use Wan VACE video inpaint instead.")
        if ensure_reference_images(kwargs.get("reference_images")):
            raise ValueError(f"{adapter.pipeline_class} does not accept reference images.")

        strength = float(kwargs.get("strength", 0.8))
        if not 0 < strength <= 1:
            raise ValueError(f"Wan {mode} strength must be greater than 0 and at most 1; received {strength}.")
        import torch

        width = int(kwargs.get("width", 832))
        height = int(kwargs.get("height", 480))
        device = getattr(pipeline, "_execution_device", None) or "cpu"
        generator = torch.Generator(device=device).manual_seed(int(kwargs.get("seed", 0)))
        call_kwargs = {
            "video": video,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_inference_steps": int(kwargs.get("num_inference_steps", 30)),
            "guidance_scale": float(kwargs.get("guidance_scale", 5.0)),
            "strength": strength,
            "num_videos_per_prompt": 1,
            "generator": generator,
            "latents": none_if_blank(kwargs.get("latents")),
            "prompt_embeds": none_if_blank(kwargs.get("prompt_embeds")),
            "negative_prompt_embeds": none_if_blank(kwargs.get("negative_prompt_embeds")),
            "output_type": kwargs.get("output_type", "pil"),
            "return_dict": True,
            "attention_kwargs": parse_json_object(kwargs.get("attention_kwargs_json"), "attention kwargs"),
            "callback_on_step_end": self.pipe_callback,
            "callback_on_step_end_tensor_inputs": callback_tensor_inputs(
                kwargs.get("callback_on_step_end_tensor_inputs")
            ),
            "max_sequence_length": int(kwargs.get("max_sequence_length", 512)),
        }
        self._active_pipeline = pipeline
        try:
            result = pipeline(**call_kwargs)
        finally:
            self._active_pipeline = None
        frames = getattr(result, "frames", result)
        if isinstance(frames, list) and len(frames) == 1 and isinstance(frames[0], list):
            frames = frames[0]
        return {
            "video_out": frames,
            "width_out": width,
            "height_out": height,
            "frames_out": len(frames) if isinstance(frames, list) else len(video),
        }

    def _execute_wan_image_to_video(
        self,
        pipeline: Any,
        adapter: VideoPipelineAdapter,
        mode: str,
        kwargs: dict[str, Any],
    ):
        if mode not in {"image_to_video", "reference_to_video"}:
            raise ValueError(f"{adapter.pipeline_class} does not support video mode {mode}.")
        if ensure_video_list(kwargs.get("video"), "video") is not None:
            raise ValueError("Wan image-to-video does not accept a source video.")
        if ensure_video_list(kwargs.get("mask"), "mask") is not None:
            raise ValueError("Wan image-to-video does not accept a mask; use Wan VACE for masked video editing.")
        references = ensure_reference_images(kwargs.get("reference_images"))
        if not references or len(references) != 1:
            raise ValueError("Wan image-to-video needs exactly one opening reference image.")

        prompt = ensure_single_prompt(none_if_blank(kwargs.get("prompt")), "prompt")
        negative_prompt = ensure_single_prompt(none_if_blank(kwargs.get("negative_prompt")), "negative prompt")
        _validate_prompt_token_limit(pipeline, prompt, "prompt", adapter.max_prompt_tokens)
        _validate_prompt_token_limit(pipeline, negative_prompt, "negative prompt", adapter.max_prompt_tokens)
        width = int(kwargs.get("width") or 832)
        height = int(kwargs.get("height") or 480)
        vae_scale = int(getattr(pipeline, "vae_scale_factor_spatial", 8) or 8)
        transformer = getattr(pipeline, "transformer", None)
        patch_size = getattr(getattr(transformer, "config", None), "patch_size", (1, 2, 2))
        spatial_patch = int(patch_size[1] if isinstance(patch_size, (list, tuple)) else patch_size)
        spatial_multiple = vae_scale * spatial_patch
        if width % spatial_multiple or height % spatial_multiple:
            raise ValueError(
                f"Wan image-to-video dimensions must be divisible by {spatial_multiple}; received {width}x{height}."
            )
        num_frames = normalize_num_frames(int(kwargs.get("num_frames") or 81), pipeline)
        if num_frames < 81:
            raise ValueError("Quality-first Wan shots need at least 81 frames (about five seconds at 16 fps).")
        import torch

        device = getattr(pipeline, "_execution_device", None) or "cpu"
        generator = torch.Generator(device=device).manual_seed(int(kwargs.get("seed") or 0))
        call_kwargs = {
            "image": references[0],
            "last_image": kwargs.get("last_image"),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": int(kwargs.get("num_inference_steps") or 40),
            "guidance_scale": float(_value_or_default(kwargs, "guidance_scale", 3.5)),
            "guidance_scale_2": float(_value_or_default(kwargs, "secondary_guidance_scale", 3.5)),
            "num_videos_per_prompt": 1,
            "generator": generator,
            "latents": none_if_blank(kwargs.get("latents")),
            "prompt_embeds": none_if_blank(kwargs.get("prompt_embeds")),
            "negative_prompt_embeds": none_if_blank(kwargs.get("negative_prompt_embeds")),
            "output_type": kwargs.get("output_type") or "pil",
            "return_dict": True,
            "attention_kwargs": parse_json_object(kwargs.get("attention_kwargs_json"), "attention kwargs"),
            "callback_on_step_end": self.pipe_callback,
            "callback_on_step_end_tensor_inputs": callback_tensor_inputs(
                kwargs.get("callback_on_step_end_tensor_inputs")
            ),
            "max_sequence_length": min(int(kwargs.get("max_sequence_length") or 512), 512),
        }
        self._active_pipeline = pipeline
        try:
            result = pipeline(**call_kwargs)
        finally:
            self._active_pipeline = None
        frames = getattr(result, "frames", result)
        if isinstance(frames, list) and len(frames) == 1 and isinstance(frames[0], list):
            frames = frames[0]
        return {
            "video_out": frames,
            "width_out": width,
            "height_out": height,
            "frames_out": len(frames) if isinstance(frames, list) else num_frames,
        }

    def _execute_ltx(self, pipeline: Any, adapter: VideoPipelineAdapter, mode: str, kwargs: dict[str, Any]):
        prompt = ensure_single_prompt(none_if_blank(kwargs.get("prompt")), "prompt")
        negative_prompt = ensure_single_prompt(none_if_blank(kwargs.get("negative_prompt")), "negative prompt")
        video = ensure_video_list(kwargs.get("video"), "video")
        reference_images = ensure_reference_images(kwargs.get("reference_images"))
        mask = ensure_video_list(kwargs.get("mask"), "mask")
        if mask is not None:
            raise ValueError(
                "LTX Video does not support the generic mask input; use a registered control adapter when available."
            )
        if mode == "text_to_video" and (video is not None or reference_images is not None):
            raise ValueError("LTX text_to_video does not accept image or video conditioning inputs.")
        if mode in {"image_to_video", "reference_to_video"} and not reference_images:
            raise ValueError(f"LTX {mode} requires at least one reference image.")
        if mode == "video_to_video" and video is None:
            raise ValueError("LTX video_to_video requires a source video.")

        _validate_prompt_token_limit(pipeline, prompt, "prompt", adapter.max_prompt_tokens)
        _validate_prompt_token_limit(pipeline, negative_prompt, "negative prompt", adapter.max_prompt_tokens)

        width = int(kwargs.get("width", 704))
        height = int(kwargs.get("height", 480))
        _validate_ltx_dimensions(width, height)
        num_frames = _normalize_ltx_frames(int(kwargs.get("num_frames", 97)))
        import torch
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

        device = getattr(pipeline, "_execution_device", None) or "cpu"
        generator = torch.Generator(device=device).manual_seed(int(kwargs.get("seed", 0)))
        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "frame_rate": int(kwargs.get("frame_rate", 25)),
            "num_inference_steps": int(kwargs.get("num_inference_steps", 40)),
            "guidance_scale": float(kwargs.get("guidance_scale", 3.0)),
            "num_videos_per_prompt": 1,
            "generator": generator,
            "latents": none_if_blank(kwargs.get("latents")),
            "prompt_embeds": none_if_blank(kwargs.get("prompt_embeds")),
            "negative_prompt_embeds": none_if_blank(kwargs.get("negative_prompt_embeds")),
            "output_type": kwargs.get("output_type", "pil"),
            "return_dict": True,
            "attention_kwargs": parse_json_object(kwargs.get("attention_kwargs_json"), "attention kwargs"),
            "callback_on_step_end": self.pipe_callback,
            "callback_on_step_end_tensor_inputs": callback_tensor_inputs(
                kwargs.get("callback_on_step_end_tensor_inputs")
            ),
            "max_sequence_length": min(
                int(kwargs.get("max_sequence_length", adapter.max_prompt_tokens or 256)),
                adapter.max_prompt_tokens or 256,
            ),
        }
        model_repo = str(getattr(pipeline, "_modiff_video_repo", "")).lower()
        inference_steps = int(call_kwargs["num_inference_steps"])
        if "distilled" in model_repo:
            if inference_steps != 8:
                raise ValueError(
                    "The LTX distilled artifact requires exactly 8 inference steps; "
                    f"received {inference_steps}. Select an LTX dev artifact for longer schedules."
                )
            if float(call_kwargs["guidance_scale"]) != 1.0:
                raise ValueError("The LTX distilled artifact requires guidance scale 1 (CFG is not used).")
            call_kwargs["negative_prompt"] = None
            call_kwargs["timesteps"] = list(LTX_DISTILLED_TIMESTEPS)
        if mode in {"image_to_video", "reference_to_video"}:
            condition_strength = float(kwargs.get("strength", 1.0))
            if len(reference_images) == 1:
                # The qualified 0.9.8 distilled checkpoint is stable when an
                # opening still is encoded as a one-frame video condition.
                # Its direct image-condition path can decode black frames for
                # partial guide strengths, so retain the validated contract.
                call_kwargs["conditions"] = [
                    LTXVideoCondition(video=[reference_images[0]], frame_index=0, strength=condition_strength)
                ]
            else:
                call_kwargs["conditions"] = [
                    LTXVideoCondition(
                        image=image,
                        frame_index=round(index * (num_frames - 1) / (len(reference_images) - 1)),
                        strength=condition_strength,
                    )
                    for index, image in enumerate(reference_images)
                ]
        elif mode == "video_to_video":
            call_kwargs["conditions"] = [
                LTXVideoCondition(
                    video=video,
                    frame_index=0,
                    strength=float(kwargs.get("strength", 1.0)),
                )
            ]
            call_kwargs["denoise_strength"] = float(kwargs.get("denoise_strength", 1.0))

        scheduler_patch = _install_ltx_dynamic_shift(pipeline, width, height, num_frames)
        self._active_pipeline = pipeline
        try:
            result = pipeline(**call_kwargs)
        finally:
            if scheduler_patch is not None:
                scheduler, original_set_timesteps = scheduler_patch
                scheduler.set_timesteps = original_set_timesteps
            self._active_pipeline = None
        frames = getattr(result, "frames", result)
        if isinstance(frames, list) and len(frames) == 1 and isinstance(frames[0], list):
            frames = frames[0]
        return {
            "video_out": frames,
            "width_out": width,
            "height_out": height,
            "frames_out": len(frames) if isinstance(frames, list) else num_frames,
        }

    def _execute_ltx_long(self, pipeline: Any, adapter: VideoPipelineAdapter, mode: str, kwargs: dict[str, Any]):
        if mode != "image_to_video":
            raise ValueError("LTX long sliding-window generation requires image_to_video mode.")
        if (
            ensure_video_list(kwargs.get("video"), "video") is not None
            or ensure_video_list(kwargs.get("mask"), "mask") is not None
        ):
            raise ValueError("LTX long image-to-video does not accept source video or mask inputs.")
        references = ensure_reference_images(kwargs.get("reference_images"))
        if not references or len(references) != 1:
            raise ValueError("LTX long image-to-video needs exactly one opening reference image.")
        prompt = ensure_single_prompt(none_if_blank(kwargs.get("prompt")), "prompt")
        _validate_prompt_token_limit(pipeline, prompt, "prompt", adapter.max_prompt_tokens)
        width = int(kwargs.get("width") or 1216)
        height = int(kwargs.get("height") or 704)
        _validate_ltx_dimensions(width, height)
        num_frames = _normalize_ltx_frames(int(kwargs.get("num_frames") or 753))
        steps = int(kwargs.get("num_inference_steps") or 8)
        guidance = float(_value_or_default(kwargs, "guidance_scale", 1))
        model_repo = str(getattr(pipeline, "_modiff_video_repo", "")).lower()
        if "distilled" in model_repo and (steps != 8 or guidance != 1):
            raise ValueError("The LTX distilled long-video pipeline requires 8 steps and guidance scale 1.")
        tile_size = int(kwargs.get("temporal_tile_size") or 80)
        overlap = int(kwargs.get("temporal_overlap") or 24)
        if overlap >= tile_size:
            raise ValueError("LTX long temporal overlap must be smaller than its temporal window.")
        segments_text = str(kwargs.get("prompt_segments_json") or "").strip()
        prompt_segments = None
        if segments_text:
            try:
                prompt_segments = json.loads(segments_text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Timed prompt segments must be valid JSON: {exc.msg}.") from exc
            if not isinstance(prompt_segments, list) or any(not isinstance(item, dict) for item in prompt_segments):
                raise ValueError("Timed prompt segments must be a JSON list of objects.")
        self._active_pipeline = pipeline
        try:
            result = pipeline(
                prompt=prompt,
                negative_prompt=None if guidance == 1 else none_if_blank(kwargs.get("negative_prompt")),
                prompt_segments=prompt_segments,
                cond_image=references[0],
                cond_strength=float(kwargs.get("strength") if kwargs.get("strength") is not None else 0.5),
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=float(kwargs.get("frame_rate") or 25),
                guidance_scale=guidance,
                num_inference_steps=steps,
                seed=int(kwargs.get("seed") or 0),
                temporal_tile_size=tile_size,
                temporal_overlap=overlap,
                temporal_overlap_cond_strength=float(
                    kwargs.get("temporal_overlap_condition_strength")
                    if kwargs.get("temporal_overlap_condition_strength") is not None
                    else 0.5
                ),
                adain_factor=float(kwargs.get("adain_factor") if kwargs.get("adain_factor") is not None else 0.25),
                decode_timestep=0.05,
                decode_noise_scale=0.025,
                output_type=kwargs.get("output_type") or "pil",
                return_dict=True,
                attention_kwargs=parse_json_object(kwargs.get("attention_kwargs_json"), "attention kwargs"),
                callback_on_step_end=self.pipe_callback,
                # The long pipeline keeps its global unpacked tensor in a local
                # named ``latents`` while denoising ``latents_packed``. Asking
                # for the generic "latents" callback input makes Diffusers
                # replace the packed sampler state with that 5D global tensor
                # on the next step. Progress/cancellation needs no tensor copy.
                callback_on_step_end_tensor_inputs=[],
                max_sequence_length=min(int(kwargs.get("max_sequence_length") or 128), 128),
            )
        finally:
            self._active_pipeline = None
        frames = getattr(result, "frames", result)
        if isinstance(frames, list) and len(frames) == 1 and isinstance(frames[0], list):
            frames = frames[0]
        return {
            "video_out": frames,
            "width_out": width,
            "height_out": height,
            "frames_out": len(frames) if isinstance(frames, list) else num_frames,
        }

    def _execute_ltx2(self, pipeline: Any, adapter: VideoPipelineAdapter, mode: str, kwargs: dict[str, Any]):
        prompt = ensure_single_prompt(none_if_blank(kwargs.get("prompt")), "prompt")
        negative_prompt = ensure_single_prompt(none_if_blank(kwargs.get("negative_prompt")), "negative prompt")
        video = ensure_video_list(kwargs.get("video"), "video")
        references = ensure_reference_images(kwargs.get("reference_images"))
        if mode == "text_to_video" and (video is not None or references is not None):
            raise ValueError("LTX-2 text_to_video does not accept image or video conditions.")
        if mode in {"image_to_video", "reference_to_video"} and not references:
            raise ValueError(f"LTX-2 {mode} requires at least one reference image.")
        if mode == "video_to_video" and not video:
            raise ValueError("LTX-2 video_to_video requires a source video.")
        width = int(kwargs.get("width") or 768)
        height = int(kwargs.get("height") or 512)
        _validate_ltx_dimensions(width, height)
        num_frames = _normalize_ltx_frames(int(kwargs.get("num_frames") or 121))
        strength = float(_value_or_default(kwargs, "strength", 1))
        import torch
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

        conditions = None
        if references:
            conditions = [
                LTX2VideoCondition(
                    frames=image,
                    index=round(index * (num_frames - 1) / max(1, len(references) - 1)),
                    strength=strength,
                )
                for index, image in enumerate(references)
            ]
        elif video:
            conditions = [LTX2VideoCondition(frames=video, index=0, strength=strength)]
        device = getattr(pipeline, "_execution_device", None) or "cpu"
        generator = torch.Generator(device=device).manual_seed(int(kwargs.get("seed") or 0))
        self._active_pipeline = pipeline
        try:
            result = pipeline(
                conditions=conditions,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=float(kwargs.get("frame_rate") or 24),
                num_inference_steps=int(kwargs.get("num_inference_steps") or 40),
                guidance_scale=float(_value_or_default(kwargs, "guidance_scale", 4)),
                generator=generator,
                output_type=kwargs.get("output_type") or "pil",
                return_dict=True,
                attention_kwargs=parse_json_object(kwargs.get("attention_kwargs_json"), "attention kwargs"),
                callback_on_step_end=self.pipe_callback,
                callback_on_step_end_tensor_inputs=callback_tensor_inputs(
                    kwargs.get("callback_on_step_end_tensor_inputs")
                ),
                max_sequence_length=min(int(kwargs.get("max_sequence_length") or 1024), 1024),
            )
        finally:
            self._active_pipeline = None
        frames = getattr(result, "frames", result)
        if isinstance(frames, list) and len(frames) == 1 and isinstance(frames[0], list):
            frames = frames[0]
        return {
            "video_out": frames,
            "width_out": width,
            "height_out": height,
            "frames_out": len(frames) if isinstance(frames, list) else num_frames,
            "_audio": getattr(result, "audios", None),
        }


class GenerateVideoAudio(NodeBase):
    """Generate synchronized video and audio from any capable Diffusers video pipeline."""

    label = "Diffusers Video + Audio Generate"
    category = "Diffusers Video"
    resizable = True
    params = {
        **Generate.params,
        "audio": {"label": "Audio", "display": "output", "type": "audio"},
        "sample_rate_out": {"label": "Sample Rate", "display": "output", "type": "int"},
        "duration_seconds": {"label": "Audio Duration", "display": "output", "type": "float"},
    }

    def __call__(self, **kwargs):
        values = dict(kwargs)
        values["mode"] = _require_video_mode(values.get("mode"))
        return super().__call__(**values)

    def execute(self, **kwargs):
        from modules.DiffusersAudio.main import output_to_audio_object

        pipeline = kwargs.get("pipeline")
        if pipeline is None:
            raise ValueError("A Diffusers video pipeline is required.")
        adapter = _pipeline_adapter(pipeline)
        if "audio" not in adapter.output_media:
            raise ValueError(
                f"{adapter.pipeline_class} does not produce synchronized audio. "
                "Use Diffusers Video Generate for video-only pipelines."
            )
        worker = Generate(self.node_id)
        worker._sid = self._sid
        worker.pipe_callback = self.pipe_callback
        executed_adapter, result = worker._execute_with_adapter(**kwargs)
        raw_audio = result.pop("_audio", None)
        if raw_audio is None:
            raise RuntimeError("The selected video pipeline did not return an audio stream.")
        vocoder = getattr(pipeline, "vocoder", None)
        vocoder_config = getattr(vocoder, "config", None)
        configured_rate = vocoder_config.get("output_sampling_rate") if hasattr(vocoder_config, "get") else None
        sample_rate = int(configured_rate or executed_adapter.default_audio_sample_rate or 48000)
        audio = output_to_audio_object(raw_audio, sample_rate=sample_rate)
        return {
            **result,
            "audio": audio,
            "sample_rate_out": sample_rate,
            "duration_seconds": float(audio["duration_seconds"]),
        }


class GenerateLTX2(GenerateVideoAudio):
    """Deprecated action alias for saved LTX-2 video-and-audio workflows."""


class BuildShotJobs(NodeBase):
    """Pair planned shots with keyframes and quality settings for a visual collection loop."""

    label = "Build Quality Video Shot Jobs"
    category = "Diffusers Video"
    resizable = True
    params = {
        "shots": {"label": "Shot Plan", "display": "input", "type": "collection"},
        "opening_images": {"label": "Opening Keyframes", "display": "input", "type": "image"},
        "ending_images": {
            "label": "Optional Ending Keyframes",
            "display": "input",
            "type": "image",
            "required": False,
        },
        "mode": {
            "label": "Shot Mode",
            "type": "string",
            "options": ["image_to_video", "text_to_video"],
            "default": "image_to_video",
        },
        "reference_policy": {
            "label": "Keyframe Pairing",
            "type": "string",
            "options": ["one_per_shot", "reuse_first", "none"],
            "default": "one_per_shot",
        },
        "negative_prompt": {
            "label": "Shared Negative Prompt",
            "display": "textarea",
            "type": "text",
            "default": "over-saturated, overexposed, static, blurred details, subtitles, illustration, painting, frozen frame, gray cast, worst quality, low quality, JPEG artifacts, ugly, deformed, disfigured, malformed limbs, fused fingers, extra limbs, inconsistent anatomy, flicker, temporal jitter, warped geometry, duplicate subject, abrupt camera jump",
        },
        "base_seed": {"label": "Base Seed", "type": "int", "default": 42017},
        "fps": {"label": "FPS", "type": "int", "default": 16, "min": 1, "max": 60},
        "minimum_seconds": {"label": "Minimum Shot Length", "type": "float", "default": 5, "min": 5},
        "width": {"label": "Width", "type": "int", "default": 832, "min": 16},
        "height": {"label": "Height", "type": "int", "default": 480, "min": 16},
        "steps": {"label": "Steps", "type": "int", "default": 40, "min": 1, "max": 100},
        "guidance_scale": {"label": "High-noise Guidance", "type": "float", "default": 3.5, "min": 0},
        "secondary_guidance_scale": {
            "label": "Low-noise Guidance",
            "type": "float",
            "default": 3.5,
            "min": 0,
        },
        "conditioning_strength": {
            "label": "Opening Keyframe Strength",
            "type": "float",
            "default": 0.9,
            "min": 0,
            "max": 1,
            "step": 0.05,
            "description": "Lower values allow more motion; higher values preserve the opening frame more exactly.",
        },
        "jobs": {"label": "Shot Jobs", "display": "output", "type": "collection"},
        "count": {"label": "Shot Count", "display": "output", "type": "int"},
        "planned_duration_seconds": {"label": "Planned Duration", "display": "output", "type": "float"},
    }

    def execute(self, **kwargs):
        shots = kwargs.get("shots")
        if not isinstance(shots, (list, tuple)) or not shots:
            raise ValueError("Build Quality Video Shot Jobs needs a non-empty shot collection.")
        openings = kwargs.get("opening_images")
        openings = (
            list(openings) if isinstance(openings, (list, tuple)) else [openings] if openings is not None else []
        )
        endings = kwargs.get("ending_images")
        endings = list(endings) if isinstance(endings, (list, tuple)) else [endings] if endings is not None else []
        mode = str(kwargs.get("mode") or "image_to_video")
        if mode not in {"image_to_video", "text_to_video"}:
            raise ValueError(f"Unsupported quality shot mode {mode!r}.")
        if mode == "image_to_video" and not openings:
            raise ValueError("Build Quality Video Shot Jobs needs opening keyframes.")
        policy = "none" if mode == "text_to_video" else str(kwargs.get("reference_policy") or "one_per_shot")
        if policy == "one_per_shot" and len(openings) != len(shots):
            raise ValueError(
                f"One-per-shot keyframe pairing needs {len(shots)} opening images; received {len(openings)}."
            )
        if policy not in {"one_per_shot", "reuse_first", "none"}:
            raise ValueError(f"Unsupported keyframe pairing policy {policy!r}.")
        if endings and len(endings) not in {1, len(shots)}:
            raise ValueError("Ending keyframes must contain one shared image or one image per shot.")

        fps = max(1, int(kwargs.get("fps") or 16))
        minimum = max(5.0, float(kwargs.get("minimum_seconds") or 5))
        base_seed = int(kwargs.get("base_seed") or 0)
        jobs = []
        planned_duration = 0.0
        for index, shot in enumerate(shots):
            if not isinstance(shot, dict) or not str(shot.get("prompt") or "").strip():
                raise ValueError(f"Shot {index + 1} needs a non-empty prompt record.")
            duration = float(shot.get("duration_seconds") or 0)
            if duration < minimum:
                raise ValueError(
                    f"Shot {index + 1} is {duration:g} seconds; quality video shots must be at least {minimum:g} seconds."
                )
            target_frames = max(1, round(duration * fps))
            num_frames = 1 + ((target_frames - 1 + 3) // 4) * 4
            opening = openings[index] if policy == "one_per_shot" else openings[0] if policy == "reuse_first" else None
            ending = endings[index] if len(endings) == len(shots) else endings[0] if endings else None
            jobs.append(
                {
                    **shot,
                    "index": index,
                    "mode": mode,
                    "opening_image": opening,
                    "ending_image": ending,
                    "negative_prompt": str(kwargs.get("negative_prompt") or "").strip(),
                    "seed": int(shot.get("seed", base_seed + index)),
                    "fps": fps,
                    "num_frames": num_frames,
                    "width": int(kwargs.get("width") or 832),
                    "height": int(kwargs.get("height") or 480),
                    "steps": int(kwargs.get("steps") or 40),
                    "guidance_scale": float(_value_or_default(kwargs, "guidance_scale", 3.5)),
                    "secondary_guidance_scale": float(_value_or_default(kwargs, "secondary_guidance_scale", 3.5)),
                    "conditioning_strength": float(
                        _value_or_default(
                            shot,
                            "conditioning_strength",
                            _value_or_default(kwargs, "conditioning_strength", 0.9),
                        )
                    ),
                }
            )
            planned_duration += num_frames / fps
        return {"jobs": jobs, "count": len(jobs), "planned_duration_seconds": planned_duration}


class GenerateShotJob(NodeBase):
    """Execute one normalized shot job through any compatible Diffusers video pipeline."""

    label = "Generate Video Shot Job"
    category = "Diffusers Video"
    params = {
        "pipeline": {
            "label": "Pipeline",
            "display": "input",
            "type": "video_diffusion_pipeline",
            "required": True,
        },
        "job": {"label": "Shot Job", "display": "input", "type": "any", "required": True},
        "video_out": {"label": "Video", "display": "output", "type": "video"},
        "width_out": {"label": "Width", "display": "output", "type": "int"},
        "height_out": {"label": "Height", "display": "output", "type": "int"},
        "frames_out": {"label": "Frames", "display": "output", "type": "int"},
        "fps_out": {"label": "FPS", "display": "output", "type": "float"},
    }

    def execute(self, **kwargs):
        pipeline = kwargs.get("pipeline")
        job = kwargs.get("job")
        if pipeline is None:
            raise ValueError("Generate Video Shot Job needs a Diffusers video pipeline.")
        if not isinstance(job, dict):
            raise TypeError("Generate Video Shot Job needs a shot job record.")
        prompt = str(job.get("prompt") or "").strip()
        opening = job.get("opening_image")
        mode = str(job.get("mode") or "image_to_video")
        if not prompt or (mode == "image_to_video" and opening is None):
            raise ValueError("A video shot job needs a prompt and image-to-video jobs also need opening_image.")

        worker = Generate(self.node_id)
        worker._sid = self._sid
        worker.pipe_callback = self.pipe_callback
        result = worker.execute(
            pipeline=pipeline,
            mode=mode,
            reference_images=[opening] if opening is not None else None,
            last_image=job.get("ending_image"),
            prompt=prompt,
            negative_prompt=str(job.get("negative_prompt") or ""),
            width=int(job.get("width") or 832),
            height=int(job.get("height") or 480),
            num_frames=int(job.get("num_frames") or 81),
            frame_rate=float(job.get("fps") or 16),
            num_inference_steps=int(job.get("steps") or 40),
            guidance_scale=float(_value_or_default(job, "guidance_scale", 3.5)),
            secondary_guidance_scale=float(_value_or_default(job, "secondary_guidance_scale", 3.5)),
            strength=float(_value_or_default(job, "conditioning_strength", 0.9)),
            seed=int(job.get("seed") or 0),
            output_type=str(job.get("output_type") or "pil"),
        )
        result["width_out"] = int(result.get("width_out") or job.get("width") or 832)
        result["height_out"] = int(result.get("height_out") or job.get("height") or 480)
        result["fps_out"] = float(job.get("fps") or 16)
        return result


class GenerateSequence(NodeBase):
    """Generate a storyboard sequence through any compatible text-to-video adapter."""

    label = "Diffusers Video Generate Sequence"
    category = "Diffusers Video"
    params = {
        **Generate.params,
        "prompts_json": {
            "label": "Shot prompts (JSON)",
            "display": "textarea",
            "type": "text",
            "default": "[]",
            "description": "JSON array of two to six prompt strings or {prompt, seed} shot objects.",
        },
        "clips": {"label": "Clips", "display": "output", "type": "video_collection"},
        "clip_count": {"label": "Clip count", "display": "output", "type": "int"},
        "total_frames": {"label": "Total frames", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        import json

        pipeline = kwargs.get("pipeline")
        if pipeline is None:
            raise ValueError("A Diffusers video pipeline is required.")
        adapter = _pipeline_adapter(pipeline)
        if "text_to_video" not in adapter.modes:
            raise ValueError(f"{adapter.pipeline_class} cannot generate a text-to-video sequence.")
        try:
            prompts = json.loads(str(kwargs.get("prompts_json") or "[]"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Shot prompts must be valid JSON: {exc.msg}.") from exc

        def normalize_shot(item, index):
            if isinstance(item, str) and item.strip():
                return {"prompt": item.strip(), "seed": None}
            if isinstance(item, dict) and isinstance(item.get("prompt"), str) and item["prompt"].strip():
                seed = item.get("seed")
                if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
                    raise ValueError(f"Shot {index + 1} seed must be an integer when supplied.")
                return {"prompt": item["prompt"].strip(), "seed": seed}
            raise ValueError(f"Shot {index + 1} must be a non-empty prompt string or a prompt object.")

        if not isinstance(prompts, list) or not 2 <= len(prompts) <= 6:
            raise ValueError("Shot prompts must be a JSON array containing two to six shots.")
        shots = [normalize_shot(item, index) for index, item in enumerate(prompts)]

        base_seed = int(kwargs.get("seed", 0))
        clips = []
        total_frames = 0
        width_out = 0
        height_out = 0
        shot_generator = Generate(self.node_id)
        shot_generator._sid = self._sid
        for index, shot in enumerate(shots):
            self.progress(
                int(index / len(shots) * 100),
                phase="sequence",
                message=f"Generating shot {index + 1} of {len(shots)}",
            )

            # Preserve interruption, timeout and Diffusers callback behavior,
            # then remap the current shot's denoising percentage onto the full
            # sequence. Without this wrapper every new shot made progress jump
            # back to zero, which was especially misleading for 30-second jobs.
            def sequence_pipe_callback(pipe, step_index, timestep, callback_kwargs, *, shot_index=index):
                result = self.pipe_callback(pipe, step_index, timestep, callback_kwargs)
                shot_steps = max(1, int(pipe._num_timesteps))
                completed_in_shot = step_index + 1
                total_steps = len(shots) * shot_steps
                completed_steps = shot_index * shot_steps + completed_in_shot
                self.progress(
                    int(completed_steps / total_steps * 100),
                    phase="denoising",
                    message=(f"Shot {shot_index + 1}/{len(shots)}: denoising {completed_in_shot}/{shot_steps}"),
                    current_step=completed_steps,
                    total_steps=total_steps,
                )
                return result

            shot_generator.pipe_callback = sequence_pipe_callback
            values = dict(
                kwargs,
                mode="text_to_video",
                prompt=shot["prompt"],
                seed=shot["seed"] if shot["seed"] is not None else base_seed + index,
            )
            result = shot_generator.execute(**values)
            clip = result["video_out"]
            clips.append(clip)
            total_frames += int(result.get("frames_out") or len(clip))
            width_out = int(result.get("width_out") or width_out)
            height_out = int(result.get("height_out") or height_out)
        self.progress(100, phase="sequence", message=f"Generated {len(shots)} of {len(shots)} shots")
        return {
            "video_out": [frame for clip in clips for frame in clip],
            "width_out": width_out,
            "height_out": height_out,
            "frames_out": total_frames,
            "clips": clips,
            "clip_count": len(clips),
            "total_frames": total_frames,
        }


class PlanLongVideo(NodeBase):
    """Create loop-ready jobs for a continuous take or an authored multi-shot fallback."""

    label = "Plan Long Video"
    category = "Diffusers Video"
    resizable = True
    params = {
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "target_seconds": {"label": "Approximate Duration", "type": "float", "default": 30, "min": 1, "max": 600},
        "fps": {"label": "FPS", "type": "int", "default": 16, "min": 1, "max": 60},
        "strategy": {
            "label": "Strategy",
            "type": "string",
            "options": ["framepack_continuous", "ltx_continuation", "wan_continuation", "multi_shot"],
            "default": "ltx_continuation",
        },
        "chunk_seconds": {"label": "Chunk Duration", "type": "float", "default": 5, "min": 1, "max": 30},
        "overlap_seconds": {"label": "Boundary Overlap", "type": "float", "default": 0.25, "min": 0, "max": 5},
        "shot_prompts": {
            "label": "Optional Shot Prompts (JSON)",
            "display": "textarea",
            "type": "text",
            "default": "[]",
        },
        "seed": {"label": "Seed", "type": "int", "default": 0},
        "jobs": {"label": "Generation Jobs", "display": "output", "type": "collection"},
        "job_count": {"label": "Jobs", "display": "output", "type": "int"},
        "planned_frames": {"label": "Planned Frames", "display": "output", "type": "int"},
        "planned_seconds": {"label": "Planned Duration", "display": "output", "type": "float"},
    }

    @staticmethod
    def _legal_frames(strategy, value):
        value = max(1, int(value))
        if strategy == "ltx_continuation":
            return _normalize_ltx_frames(value)
        if strategy == "wan_continuation":
            remainder = (value - 1) % 4
            return value if remainder == 0 else value + 4 - remainder
        return value

    def execute(self, **kwargs):
        import json
        from math import ceil

        prompt = str(kwargs.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Plan Long Video needs a prompt.")
        strategy = str(kwargs.get("strategy") or "ltx_continuation")
        fps = max(1, int(kwargs.get("fps") or 16))
        target_frames = max(1, round(float(kwargs.get("target_seconds") or 30) * fps))
        seed = int(kwargs.get("seed") or 0)
        raw_shots = kwargs.get("shot_prompts") or "[]"
        try:
            shot_prompts = json.loads(raw_shots) if isinstance(raw_shots, str) else raw_shots
        except json.JSONDecodeError as exc:
            raise ValueError(f"Shot prompts must be valid JSON: {exc.msg}.") from exc
        if not isinstance(shot_prompts, list) or any(
            not isinstance(item, str) or not item.strip() for item in shot_prompts
        ):
            raise ValueError("Shot prompts must be a JSON array of non-empty strings.")

        if strategy == "framepack_continuous":
            chunk_frames = target_frames
            count = 1
        else:
            chunk_frames = self._legal_frames(strategy, round(float(kwargs.get("chunk_seconds") or 5) * fps))
            overlap = min(
                max(0, round(float(kwargs.get("overlap_seconds") or 0) * fps)),
                max(0, chunk_frames - 1),
            )
            stride = max(1, chunk_frames - overlap)
            count = max(1, ceil(max(0, target_frames - overlap) / stride))
        jobs = []
        for index in range(count):
            authored = shot_prompts[index % len(shot_prompts)].strip() if shot_prompts else prompt
            continuity = index > 0 and strategy in {"ltx_continuation", "wan_continuation"}
            jobs.append(
                {
                    "index": index,
                    "prompt": authored,
                    "seed": seed + index,
                    "num_frames": chunk_frames,
                    "mode": "image_to_video" if strategy != "multi_shot" else "text_to_video",
                    "uses_previous_last_frame": continuity,
                    "strategy": strategy,
                }
            )
        overlap_frames = (
            0
            if count == 1
            else min(
                max(0, round(float(kwargs.get("overlap_seconds") or 0) * fps)),
                max(0, chunk_frames - 1),
            )
        )
        planned_frames = chunk_frames * count - overlap_frames * max(0, count - 1)
        return {
            "jobs": jobs,
            "job_count": len(jobs),
            "planned_frames": planned_frames,
            "planned_seconds": planned_frames / fps,
        }
