"""Model-neutral Diffusers video facade nodes.

Node identity describes the media contract; registered adapters own pipeline
loading, input validation, and family-specific argument translation.  Legacy
Wan node keys remain registered separately for persisted workflows.
"""

from dataclasses import dataclass
from functools import wraps
import json
import logging
from typing import Any

from modiff.config import CONFIG
from modiff.NodeBase import NodeBase
from modiff.diffusers_offload import OFFLOAD_MODE_MODEL_CPU, apply_pipeline_offload
from modiff.model_artifact_catalog import require_catalog_revision, resolve_model_revision
from modules.DiffusersVideo.wan_vace import (
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
)
from utils.huggingface import local_files_only
from utils.torch_utils import DEFAULT_DEVICE, str_to_dtype

logger = logging.getLogger("modiff")

# LTX's distilled 13B release is trained for this non-uniform eight-evaluation
# trajectory.  Diffusers otherwise derives a generic linear-quadratic schedule,
# which is appropriate for the dev checkpoint but not the distilled artifact.
LTX_DISTILLED_TIMESTEPS = [1000, 900, 700, 500, 300, 200, 100, 40]
FRAMEPACK_BASE_REPO = "hunyuanvideo-community/HunyuanVideo"
FRAMEPACK_VISION_REPO = "lllyasviel/flux_redux_bfl"


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


VIDEO_PIPELINE_ADAPTERS = {
    "WanVACEPipeline": VideoPipelineAdapter(
        id="wan-vace",
        pipeline_class="WanVACEPipeline",
        diffusers_class="WanVACEPipeline",
        default_repo="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        modes=(
            "text_to_video",
            "image_to_video",
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


def get_video_pipeline_adapter(name: Any) -> VideoPipelineAdapter:
    key = str(name or "WanVACEPipeline")
    adapter = VIDEO_PIPELINE_ADAPTERS.get(key)
    if adapter is None:
        supported = ", ".join(sorted(VIDEO_PIPELINE_ADAPTERS))
        raise ValueError(f"Unsupported Diffusers video pipeline class {key}. Supported classes: {supported}.")
    return adapter


def _resolve_adapter_model_selection(adapter: VideoPipelineAdapter, value: Any):
    """Replace only the inherited Wan VACE default for non-VACE adapters.

    ``LoadPipeline`` intentionally inherits the mature Wan loader contract, so
    its model selector also inherits Wan's persisted default value.  A graph
    that changes only ``pipeline_class`` must resolve to that adapter's model;
    an explicitly selected local or Hub artifact must remain untouched.
    """

    selected = repo_value(value)
    inherited_vace_default = VIDEO_PIPELINE_ADAPTERS["WanVACEPipeline"].default_repo
    if not selected or (adapter.pipeline_class != "WanVACEPipeline" and selected == inherited_vace_default):
        return {"source": "hub", "value": adapter.default_repo}
    return value


def _resolve_loader_revision(model_selection: Any, model_id: str, revision: Any) -> str | None:
    source = model_selection.get("source") if isinstance(model_selection, dict) else None
    return resolve_model_revision(model_id, none_if_blank(revision), source=source)


def _pipeline_adapter(pipeline: Any) -> VideoPipelineAdapter:
    adapter_name = getattr(pipeline, "_modiff_video_pipeline_class", None)
    if adapter_name:
        return get_video_pipeline_adapter(adapter_name)
    # Compatibility for pipelines loaded before adapter tagging existed.
    return get_video_pipeline_adapter("WanVACEPipeline")


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
        "pipeline": {"label": "Pipeline", "display": "output", "type": "video_diffusion_pipeline"},
        "pipeline_class": {
            "label": "Pipeline Class",
            "type": "string",
            "options": list(VIDEO_PIPELINE_ADAPTERS),
            "default": "WanVACEPipeline",
        },
        "resolved_artifact": {"label": "Resolved Artifact", "display": "output", "type": "string"},
    }

    def execute(self, **kwargs):
        adapter = get_video_pipeline_adapter(kwargs.get("pipeline_class"))
        values = dict(kwargs)
        values["model_id"] = _resolve_adapter_model_selection(adapter, values.get("model_id"))
        if adapter.pipeline_class == "WanVACEPipeline":
            result = super().execute(**values)
            pipeline = result["pipeline"]
        elif adapter.pipeline_class == "WanVideoToVideoPipeline":
            pipeline = self._load_wan_video_to_video(adapter, values)
            result = {"pipeline": pipeline}
        elif adapter.pipeline_class in {"WanPipeline", "Wan22Pipeline", "WanTI2VPipeline"}:
            pipeline = self._load_wan_text_to_video(adapter, values)
            result = {"pipeline": pipeline}
        elif adapter.pipeline_class == "WanImageToVideoPipeline":
            pipeline = self._load_wan_image_to_video(adapter, values)
            result = {"pipeline": pipeline}
        elif adapter.pipeline_class == "LTXConditionPipeline":
            pipeline = self._load_ltx(adapter, values)
            result = {"pipeline": pipeline}
        elif adapter.pipeline_class == "LTXI2VLongMultiPromptPipeline":
            pipeline = self._load_ltx_long(adapter, values)
            result = {"pipeline": pipeline}
        elif adapter.pipeline_class == "LTX2ConditionPipeline":
            pipeline = self._load_ltx2(adapter, values)
            result = {"pipeline": pipeline}
        elif adapter.pipeline_class == "WanAnimatePipeline":
            pipeline = self._load_wan_animate(adapter, values)
            result = {"pipeline": pipeline}
        else:
            pipeline = self._load_framepack(adapter, values)
            result = {"pipeline": pipeline}
        setattr(pipeline, "_modiff_video_pipeline_class", adapter.pipeline_class)
        setattr(pipeline, "_modiff_video_repo", repo_value(values.get("model_id")) or adapter.default_repo)
        return {**result, "resolved_artifact": values.get("model_id")}

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
        "pipeline": {"label": "Pipeline", "display": "input", "type": "video_diffusion_pipeline", "required": True},
        "mode": {
            "label": "Mode",
            "type": "string",
            "options": sorted({mode for adapter in VIDEO_PIPELINE_ADAPTERS.values() for mode in adapter.modes}),
            "default": "text_to_video",
        },
        "frame_rate": {"label": "Frame rate", "type": "int", "default": 25, "min": 1, "max": 60},
        "strength": {"label": "Condition strength", "type": "float", "default": 1.0, "min": 0, "max": 1, "step": 0.05},
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
        "background_video": {"label": "Background Video", "display": "input", "type": ["video", "str"], "required": False},
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

    def execute(self, **kwargs):
        _adapter, result = self._execute_with_adapter(**kwargs)
        result.pop("_audio", None)
        return result

    def _execute_with_adapter(self, **kwargs):
        pipeline = kwargs.get("pipeline")
        if pipeline is None:
            raise ValueError("A Diffusers video pipeline is required.")
        adapter = _pipeline_adapter(pipeline)
        mode = str(kwargs.get("mode") or "text_to_video")
        if mode not in adapter.modes:
            raise ValueError(f"{adapter.pipeline_class} does not support video mode {mode}.")
        return adapter, self._execute_adapter(pipeline, adapter, mode, kwargs)

    def _execute_adapter(
        self,
        pipeline: Any,
        adapter: VideoPipelineAdapter,
        mode: str,
        kwargs: dict[str, Any],
    ):
        if adapter.pipeline_class == "WanVACEPipeline":
            return super().execute(**kwargs)
        if adapter.pipeline_class == "WanVideoToVideoPipeline":
            return self._execute_wan_video_to_video(pipeline, adapter, mode, kwargs)
        if adapter.pipeline_class in {"WanPipeline", "Wan22Pipeline", "WanTI2VPipeline"}:
            return self._execute_wan_text_to_video(pipeline, adapter, mode, kwargs)
        if adapter.pipeline_class == "WanImageToVideoPipeline":
            return self._execute_wan_image_to_video(pipeline, adapter, mode, kwargs)
        if adapter.pipeline_class == "LTXConditionPipeline":
            return self._execute_ltx(pipeline, adapter, mode, kwargs)
        if adapter.pipeline_class == "LTXI2VLongMultiPromptPipeline":
            return self._execute_ltx_long(pipeline, adapter, mode, kwargs)
        if adapter.pipeline_class == "LTX2ConditionPipeline":
            return self._execute_ltx2(pipeline, adapter, mode, kwargs)
        if adapter.pipeline_class == "WanAnimatePipeline":
            return self._execute_wan_animate(pipeline, adapter, mode, kwargs)
        return self._execute_framepack(pipeline, adapter, mode, kwargs)

    def _execute_wan_animate(self, pipeline: Any, adapter: VideoPipelineAdapter, mode: str, kwargs: dict[str, Any]):
        import torch

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
        import torch

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
        import torch

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
        import torch

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
        import torch

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
        import torch
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

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
        import torch
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

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
        "ending_images": {"label": "Optional Ending Keyframes", "display": "input", "type": "image", "required": False},
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
                    "secondary_guidance_scale": float(
                        _value_or_default(kwargs, "secondary_guidance_scale", 3.5)
                    ),
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
                    message=(
                        f"Shot {shot_index + 1}/{len(shots)}: "
                        f"denoising {completed_in_shot}/{shot_steps}"
                    ),
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
