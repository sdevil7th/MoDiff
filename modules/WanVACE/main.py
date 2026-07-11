import json
import logging
from typing import Any

from modiff.config import CONFIG
from modiff.NodeBase import NodeBase
from modiff.diffusers_offload import (
    OFFLOAD_MODE_MODEL_CPU,
    apply_pipeline_offload,
    normalize_offload_mode,
    offload_mode_param,
)
from utils.huggingface import local_files_only
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST, str_to_dtype

logger = logging.getLogger("modiff")

WAN_VACE_DEFAULT_REPO = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
DEVICE_OPTIONS = list(DEVICE_LIST.keys())


def repo_value(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("value") or "")
    return str(value or "")


def none_if_blank(value: Any):
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def ensure_single_prompt(prompt: Any, field_name: str):
    if isinstance(prompt, list):
        raise ValueError(f"Wan VACE currently supports one {field_name}; prompt lists are not supported by this pipeline.")
    return prompt


def ensure_video_list(value: Any, field_name: str):
    if value in (None, ""):
        return None
    if isinstance(value, list):
        return value
    raise ValueError(f"{field_name} must be a video frame list or be left empty.")


def ensure_reference_images(value: Any):
    if value in (None, ""):
        return None
    if not isinstance(value, list):
        return [value]
    return value


def parse_json_object(value: Any, field_name: str):
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must decode to a JSON object.")
    return parsed


def callback_tensor_inputs(value: Any):
    if value in (None, ""):
        return ["latents"]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def normalize_num_frames(num_frames: int, pipeline: Any) -> int:
    vae_scale = int(getattr(pipeline, "vae_scale_factor_temporal", 4) or 4)
    if num_frames < 1:
        return 1
    remainder = (num_frames - 1) % vae_scale
    return num_frames if remainder == 0 else num_frames + (vae_scale - remainder)


def validate_dimensions(width: int, height: int, pipeline: Any):
    spatial_scale = int(getattr(pipeline, "vae_scale_factor_spatial", 8) or 8)
    transformer = getattr(pipeline, "transformer", None)
    patch_size = getattr(getattr(transformer, "config", None), "patch_size", (1, 2, 2))
    patch_width = patch_size[2] if len(patch_size) > 2 else patch_size[-1]
    patch_height = patch_size[1] if len(patch_size) > 1 else patch_size[-1]
    width_multiple = spatial_scale * int(patch_width)
    height_multiple = spatial_scale * int(patch_height)
    if width % width_multiple != 0 or height % height_multiple != 0:
        raise ValueError(
            f"Wan VACE size must be divisible by {width_multiple}x{height_multiple}; "
            f"received {width}x{height}."
        )


class LoadPipeline(NodeBase):
    """Load a Wan VACE Diffusers video pipeline."""

    label = "Load Wan VACE"
    category = "Video AI"
    resizable = True
    params = {
        "pipeline": {"label": "Pipeline", "display": "output", "type": "wan_vace_pipeline"},
        "model_id": {
            "label": "Model",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": "Wan-AI/Wan2.1-VACE-1.3B-diffusers"},
            "fieldOptions": {"noValidation": True, "sources": ["hub", "local"]},
        },
        "revision": {"label": "Revision", "type": "string", "default": ""},
        "dtype": {
            "label": "DType",
            "type": "string",
            "options": ["float32", "float16", "bfloat16"],
            "default": "bfloat16",
        },
        "device": {
            "label": "Device",
            "type": "string",
            "options": DEVICE_OPTIONS,
            "default": DEFAULT_DEVICE,
        },
        "auto_offload": {"label": "Auto offload", "type": "bool", "default": True},
        "offload_mode": offload_mode_param(),
        "low_cpu_mem_usage": {"label": "Low CPU memory", "type": "bool", "default": True},
    }

    def execute(self, **kwargs):
        import torch
        from diffusers import AutoencoderKLWan, WanVACEPipeline

        model_id = repo_value(kwargs.get("model_id")) or WAN_VACE_DEFAULT_REPO
        dtype = str_to_dtype(kwargs.get("dtype", "bfloat16"))
        revision = none_if_blank(kwargs.get("revision"))
        device = kwargs.get("device") or DEFAULT_DEVICE
        auto_offload = bool(kwargs.get("auto_offload", True))
        offload_mode = normalize_offload_mode(kwargs.get("offload_mode") or OFFLOAD_MODE_MODEL_CPU, auto_offload=auto_offload)

        logger.info("Loading Wan VACE pipeline: %s", model_id)
        self.progress(-1, phase="loading", message="Loading Wan VACE pipeline")
        offline = local_files_only(model_id)
        component_load_kwargs = {
            "revision": revision,
            "local_files_only": offline,
        }
        if CONFIG.hf.get("cache_dir"):
            component_load_kwargs["cache_dir"] = CONFIG.hf["cache_dir"]

        self.progress(-1, phase="loading", message="Loading Wan VACE VAE in float32")
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            **component_load_kwargs,
        )
        load_kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": bool(kwargs.get("low_cpu_mem_usage", True)),
            "vae": vae,
            **component_load_kwargs,
        }
        pipeline = WanVACEPipeline.from_pretrained(model_id, **load_kwargs)

        self.progress(-1, phase="loading", message=f"Applying {offload_mode} offload")
        apply_pipeline_offload(
            pipeline,
            mode=offload_mode,
            device=device,
            node_id=self.node_id,
            scope="wan-vace",
        )

        self.mm_add(pipeline, priority=2)
        return {"pipeline": pipeline}


class Generate(NodeBase):
    """Generate or edit video with the Wan VACE pipeline."""

    label = "Wan VACE Generate"
    category = "Video AI"
    resizable = True
    params = {
        "pipeline": {"label": "Pipeline", "display": "input", "type": "wan_vace_pipeline"},
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "negative_prompt": {"label": "Negative Prompt", "display": "textarea", "type": "text", "default": ""},
        "video": {"label": "Source/control video", "display": "input", "type": "video"},
        "mask": {"label": "Mask video", "display": "input", "type": "video"},
        "reference_images": {"label": "Reference images", "display": "input", "type": "image"},
        "conditioning_scale": {"label": "Conditioning Scale", "display": "slider", "type": "float", "min": 0, "max": 2, "step": 0.05, "default": 1.0},
        "width": {"label": "Width", "type": "int", "default": 832, "min": 16, "max": 2048, "step": 16},
        "height": {"label": "Height", "type": "int", "default": 480, "min": 16, "max": 2048, "step": 16},
        "num_frames": {"label": "Frames", "type": "int", "default": 81, "min": 1, "max": 241, "step": 4},
        "num_inference_steps": {"label": "Steps", "display": "slider", "type": "int", "default": 30, "min": 1, "max": 100},
        "guidance_scale": {"label": "Guidance", "display": "slider", "type": "float", "default": 5.0, "min": 0, "max": 20, "step": 0.1},
        "guidance_scale_2": {"label": "Guidance 2", "display": "slider", "type": "float", "default": 0.0, "min": 0, "max": 20, "step": 0.1},
        "use_guidance_scale_2": {"label": "Use guidance 2", "type": "bool", "default": False},
        "num_videos_per_prompt": {"label": "Videos per prompt", "type": "int", "default": 1, "min": 1, "max": 1},
        "seed": {"label": "Seed", "type": "int", "display": "random", "default": 0, "min": 0, "max": 4294967295},
        "latents": {"label": "Latents", "display": "input", "type": "tensor"},
        "prompt_embeds": {"label": "Prompt embeds", "display": "input", "type": "tensor"},
        "negative_prompt_embeds": {"label": "Negative prompt embeds", "display": "input", "type": "tensor"},
        "output_type": {"label": "Output type", "type": "string", "options": ["pil", "np", "pt"], "default": "pil"},
        "attention_kwargs_json": {"label": "Attention kwargs JSON", "display": "textarea", "type": "text", "default": ""},
        "callback_on_step_end_tensor_inputs": {"label": "Callback tensors", "type": "string", "default": "latents"},
        "max_sequence_length": {"label": "Max sequence length", "type": "int", "default": 512, "min": 1, "max": 2048},
        "video_out": {"label": "Video frames", "display": "output", "type": "video"},
        "width_out": {"label": "Width", "display": "output", "type": "int"},
        "height_out": {"label": "Height", "display": "output", "type": "int"},
        "frames_out": {"label": "Frames", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        import torch

        pipeline = kwargs.get("pipeline")
        if pipeline is None:
            raise ValueError("Wan VACE pipeline is required.")

        prompt = ensure_single_prompt(none_if_blank(kwargs.get("prompt")), "prompt")
        negative_prompt = ensure_single_prompt(none_if_blank(kwargs.get("negative_prompt")), "negative prompt")
        video = ensure_video_list(kwargs.get("video"), "video")
        mask = ensure_video_list(kwargs.get("mask"), "mask")
        reference_images = ensure_reference_images(kwargs.get("reference_images"))

        if mask is not None and video is None:
            raise ValueError("Wan VACE mask input requires a source/control video input.")
        if video is not None and mask is not None and len(video) != len(mask):
            raise ValueError(f"Wan VACE video/mask frame count mismatch: {len(video)} vs {len(mask)}.")

        num_videos_per_prompt = int(kwargs.get("num_videos_per_prompt", 1))
        if num_videos_per_prompt != 1:
            raise ValueError("Wan VACE currently supports num_videos_per_prompt=1.")

        width = int(kwargs.get("width", 832))
        height = int(kwargs.get("height", 480))
        num_frames = normalize_num_frames(int(kwargs.get("num_frames", 81)), pipeline)
        validate_dimensions(width, height, pipeline)

        device = getattr(pipeline, "_execution_device", None) or "cpu"
        generator = torch.Generator(device=device).manual_seed(int(kwargs.get("seed", 0)))

        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "video": video,
            "mask": mask,
            "reference_images": reference_images,
            "conditioning_scale": float(kwargs.get("conditioning_scale", 1.0)),
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": int(kwargs.get("num_inference_steps", 30)),
            "guidance_scale": float(kwargs.get("guidance_scale", 5.0)),
            "num_videos_per_prompt": num_videos_per_prompt,
            "generator": generator,
            "latents": none_if_blank(kwargs.get("latents")),
            "prompt_embeds": none_if_blank(kwargs.get("prompt_embeds")),
            "negative_prompt_embeds": none_if_blank(kwargs.get("negative_prompt_embeds")),
            "output_type": kwargs.get("output_type", "pil"),
            "return_dict": True,
            "attention_kwargs": parse_json_object(kwargs.get("attention_kwargs_json"), "attention kwargs"),
            "callback_on_step_end": self.pipe_callback,
            "callback_on_step_end_tensor_inputs": callback_tensor_inputs(kwargs.get("callback_on_step_end_tensor_inputs")),
            "max_sequence_length": int(kwargs.get("max_sequence_length", 512)),
        }

        if bool(kwargs.get("use_guidance_scale_2", False)):
            if getattr(pipeline, "boundary_ratio", None) is None:
                raise ValueError("guidance_scale_2 is only valid for Wan VACE pipelines with boundary_ratio support.")
            call_kwargs["guidance_scale_2"] = float(kwargs.get("guidance_scale_2", 0.0))

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
