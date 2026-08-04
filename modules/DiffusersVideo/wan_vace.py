import json
import logging
from typing import Any

from modiff.config import CONFIG
from modiff.NodeBase import NodeBase
from modiff.diffusers_offload import (
    OFFLOAD_MODE_MODEL_CPU,
    apply_pipeline_offload,
    offload_mode_param,
)
from modiff.model_artifact_catalog import resolve_model_revision
from utils.huggingface import local_files_only
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST, str_to_dtype

logger = logging.getLogger("modiff")

WAN_VACE_NATIVE_CHUNK_FRAMES = 81

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


def _black_frame_like(frame: Any):
    """Return a black conditioning mask without changing its container type."""

    try:
        import torch

        if isinstance(frame, torch.Tensor):
            return torch.zeros_like(frame)
    except ImportError:  # pragma: no cover - torch is a runtime dependency
        pass

    try:
        import numpy as np

        if isinstance(frame, np.ndarray):
            return np.zeros_like(frame)
    except ImportError:  # pragma: no cover - numpy is a runtime dependency
        pass

    if hasattr(frame, "mode") and hasattr(frame, "size"):
        from PIL import Image

        return Image.new(frame.mode, frame.size, 0)
    raise TypeError(f"Wan VACE cannot create a continuity mask for {type(frame).__name__} frames.")


def _neutralize_masked_region(frame: Any, mask: Any):
    """Replace generated VACE regions with the contract's neutral gray value.

    VACE treats white mask pixels as regions to generate and black pixels as
    source pixels to preserve.  The corresponding source pixels must be
    neutral gray; leaving the original image under a white mask can make the
    pipeline preserve the old object instead of honoring the edit.
    """

    try:
        import torch

        if isinstance(frame, torch.Tensor) and isinstance(mask, torch.Tensor):
            mask_values = mask
            threshold = 0.5 if mask_values.is_floating_point() and float(mask_values.max()) <= 1 else 127
            generate = mask_values > threshold
            while generate.ndim < frame.ndim:
                generate = generate.unsqueeze(-1)
            if generate.shape != frame.shape:
                generate = torch.broadcast_to(generate, frame.shape)
            if frame.is_floating_point():
                minimum = float(frame.min())
                maximum = float(frame.max())
                gray = 0.5 if minimum >= 0 and maximum <= 1 else (0.0 if minimum >= -1 and maximum <= 1 else 127.0)
            else:
                gray = 127
            return torch.where(generate, torch.as_tensor(gray, dtype=frame.dtype, device=frame.device), frame)
    except ImportError:  # pragma: no cover - torch is a runtime dependency
        pass

    try:
        import numpy as np

        if isinstance(frame, np.ndarray) and isinstance(mask, np.ndarray):
            mask_values = mask[..., 0] if mask.ndim == frame.ndim else mask
            threshold = 0.5 if np.issubdtype(mask_values.dtype, np.floating) and float(mask_values.max()) <= 1 else 127
            generate = mask_values > threshold
            while generate.ndim < frame.ndim:
                generate = np.expand_dims(generate, axis=-1)
            if np.issubdtype(frame.dtype, np.floating):
                minimum = float(frame.min())
                maximum = float(frame.max())
                gray = 0.5 if minimum >= 0 and maximum <= 1 else (0.0 if minimum >= -1 and maximum <= 1 else 127.0)
            else:
                gray = 127
            return np.where(generate, np.asarray(gray, dtype=frame.dtype), frame)
    except ImportError:  # pragma: no cover - numpy is a runtime dependency
        pass

    if hasattr(frame, "mode") and hasattr(frame, "size") and hasattr(mask, "convert"):
        from PIL import Image

        bands = frame.getbands()
        neutral_color = 127 if len(bands) == 1 else tuple(255 if band == "A" else 127 for band in bands)
        neutral = Image.new(frame.mode, frame.size, neutral_color)
        return Image.composite(neutral, frame, mask.convert("L"))
    raise TypeError(
        f"Wan VACE cannot neutralize {type(frame).__name__} frames with {type(mask).__name__} masks."
    )


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


class WanVACELoadPipeline(NodeBase):
    """Internal loader adapter for Hugging Face Diffusers' Wan VACE pipeline."""

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
        "execution_recipe": {
            "label": "Execution Recipe",
            "display": "input",
            "type": "diffusers_execution_recipe",
        },
        "low_cpu_mem_usage": {"label": "Low CPU memory", "type": "bool", "default": True},
    }

    def execute(self, **kwargs):
        import torch
        from diffusers import AutoencoderKLWan, WanVACEPipeline
        from modules.DiffusersRuntime.main import apply_execution_recipe_to_pipeline, loader_runtime_options

        model_selection = kwargs.get("model_id")
        selected_model_id = repo_value(model_selection)
        model_id = selected_model_id or WAN_VACE_DEFAULT_REPO
        model_source = model_selection.get("source") if selected_model_id and isinstance(model_selection, dict) else "hub"
        dtype = str_to_dtype(kwargs.get("dtype", "bfloat16"))
        revision = resolve_model_revision(
            model_id,
            none_if_blank(kwargs.get("revision")),
            source=model_source,
        )
        recipe, device, offload_mode, recipe_load_kwargs = loader_runtime_options(
            kwargs,
            default_device=DEFAULT_DEVICE,
            default_offload_mode=OFFLOAD_MODE_MODEL_CPU,
        )

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
            **recipe_load_kwargs,
        }
        pipeline = WanVACEPipeline.from_pretrained(model_id, **load_kwargs)
        apply_execution_recipe_to_pipeline(pipeline, recipe)

        self.progress(-1, phase="component_placement", message=f"Applying {offload_mode} offload")
        apply_pipeline_offload(
            pipeline,
            mode=offload_mode,
            device=device,
            node_id=self.node_id,
            scope="wan-vace",
        )

        self.mm_add(pipeline, priority=2)
        return {"pipeline": pipeline}


class WanVACEGenerate(NodeBase):
    """Internal generation adapter for Hugging Face Diffusers' Wan VACE pipeline."""

    label = "Wan VACE Generate"
    category = "Video AI"
    resizable = True
    params = {
        "pipeline": {"label": "Pipeline", "display": "input", "type": "wan_vace_pipeline", "required": True},
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "negative_prompt": {"label": "Negative Prompt", "display": "textarea", "type": "text", "default": ""},
        "video": {"label": "Source/control video", "display": "input", "type": "video", "required": False},
        "mask": {"label": "Mask video", "display": "input", "type": "video", "required": False},
        "reference_images": {"label": "Reference images", "display": "input", "type": "image", "required": False},
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
        "latents": {"label": "Latents", "display": "input", "type": "tensor", "required": False},
        "prompt_embeds": {"label": "Prompt embeds", "display": "input", "type": "tensor", "required": False},
        "negative_prompt_embeds": {"label": "Negative prompt embeds", "display": "input", "type": "tensor", "required": False},
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
        if video is not None and mask is not None:
            video = [_neutralize_masked_region(frame, mask_frame) for frame, mask_frame in zip(video, mask)]

        num_videos_per_prompt = int(kwargs.get("num_videos_per_prompt", 1))
        if num_videos_per_prompt != 1:
            raise ValueError("Wan VACE currently supports num_videos_per_prompt=1.")

        width = int(kwargs.get("width", 832))
        height = int(kwargs.get("height", 480))
        num_frames = normalize_num_frames(int(kwargs.get("num_frames", 81)), pipeline)
        validate_dimensions(width, height, pipeline)

        device = getattr(pipeline, "_execution_device", None) or "cpu"
        seed = int(kwargs.get("seed", 0))

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
            "generator": None,
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

        def run_chunk(chunk_kwargs, chunk_seed):
            values = dict(chunk_kwargs)
            values["generator"] = torch.Generator(device=device).manual_seed(chunk_seed)
            self._active_pipeline = pipeline
            try:
                result = pipeline(**values)
            finally:
                self._active_pipeline = None
            frames = getattr(result, "frames", result)
            if isinstance(frames, list) and len(frames) == 1 and isinstance(frames[0], list):
                frames = frames[0]
            return frames

        # The official VACE contract is trained and documented around native
        # ~5 second (81 frame) clips. Passing a 161-frame source directly is
        # substantially slower and, in live proofs, caused the requested mask
        # edit to be ignored. Keep the node contract model-neutral while the
        # adapter segments longer source-conditioned videos into overlapping
        # native clips. For masked edits, the prior generated final frame is a
        # black-masked continuity anchor for the next clip.
        if video is not None and num_frames > WAN_VACE_NATIVE_CHUNK_FRAMES:
            if call_kwargs["latents"] is not None:
                raise ValueError("Custom Wan VACE latents cannot be reused across a segmented long-video run.")
            frames = []
            start = 0
            chunk_index = 0
            total_chunks = (num_frames - 2) // (WAN_VACE_NATIVE_CHUNK_FRAMES - 1) + 1
            while start < num_frames:
                end = min(start + WAN_VACE_NATIVE_CHUNK_FRAMES, num_frames)
                chunk_video = list(video[start:end])
                chunk_mask = list(mask[start:end]) if mask is not None else None
                if frames and chunk_mask is not None:
                    chunk_video[0] = frames[-1]
                    chunk_mask[0] = _black_frame_like(chunk_mask[0])
                values = dict(call_kwargs)
                values["video"] = chunk_video
                values["mask"] = chunk_mask
                values["num_frames"] = len(chunk_video)
                self.progress(
                    chunk_index / total_chunks,
                    phase="sequence",
                    message=f"Generating native VACE segment {chunk_index + 1}",
                )
                # A segmented edit is one logical generation. Keep the same
                # locked seed across native chunks so material/subject identity
                # cannot reset at the continuation boundary; the generated
                # overlap frame supplies temporal position.
                chunk_frames = run_chunk(values, seed)
                if not isinstance(chunk_frames, list) or len(chunk_frames) != len(chunk_video):
                    raise RuntimeError(
                        f"Wan VACE segment {chunk_index + 1} returned "
                        f"{len(chunk_frames) if isinstance(chunk_frames, list) else 'an unknown number of'} frames; "
                        f"expected {len(chunk_video)}."
                    )
                frames.extend(chunk_frames if not frames else chunk_frames[1:])
                if end >= num_frames:
                    break
                start = end - 1
                chunk_index += 1
        else:
            frames = run_chunk(call_kwargs, seed)

        return {
            "video_out": frames,
            "width_out": width,
            "height_out": height,
            "frames_out": len(frames) if isinstance(frames, list) else num_frames,
        }
