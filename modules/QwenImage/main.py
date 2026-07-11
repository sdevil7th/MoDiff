import logging
import inspect
from typing import Any

from PIL import Image, ImageColor, ImageDraw, ImageFilter

from modiff.NodeBase import NodeBase
from modiff.diffusers_offload import (
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
    apply_pipeline_offload,
    normalize_offload_mode,
    offload_mode_param,
)
from modiff.diffusers_profiles import QWEN_IMAGE_2512_PREQUANTIZED_REPO, QWEN_IMAGE_2512_REPO
from utils.huggingface import local_files_only
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST, str_to_dtype

logger = logging.getLogger("modiff")

QWEN_IMAGE_EDIT_DEFAULT_REPO = "Qwen/Qwen-Image-Edit"
DEVICE_OPTIONS = list(DEVICE_LIST.keys())
QWEN_DIRECT_OFFLOAD_MODES = [
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
]
QWEN_QUANTIZABLE_COMPONENTS = ["transformer", "text_encoder"]


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
        raise ValueError(f"Qwen Image inpaint currently supports one {field_name}; prompt lists are not supported by this pipeline.")
    return prompt


def ensure_single_image(value: Any, field_name: str) -> Image.Image:
    if value in (None, ""):
        raise ValueError(f"Qwen Image workflow requires a {field_name}.")
    if isinstance(value, list):
        images = [item for item in value if item not in (None, "")]
        if len(images) != 1:
            raise ValueError(f"Qwen Image workflow requires exactly one {field_name}; received {len(images)}.")
        value = images[0]
    if not isinstance(value, Image.Image):
        raise ValueError(f"Qwen Image workflow {field_name} must be a PIL image loaded by the Image Load node.")
    return value


def normalize_padding_mask_crop(value: Any):
    if value in (None, ""):
        return None
    value = int(value)
    return value if value > 0 else None


def coerce_pipeline_quantization_config(quant_config: Any):
    if not quant_config:
        return None
    if isinstance(quant_config, str):
        if quant_config.strip() == "":
            return None
        raise ValueError(
            "Qwen Image quant_config must be connected to a Quantization Config node output. "
            f"Received unresolved string value {quant_config!r}."
        )

    from diffusers.quantizers import PipelineQuantizationConfig

    if isinstance(quant_config, PipelineQuantizationConfig):
        return quant_config
    if isinstance(quant_config, dict):
        return PipelineQuantizationConfig(quant_mapping=quant_config)
    raise ValueError("Qwen Image inpaint quant_config must be a Diffusers PipelineQuantizationConfig or component mapping.")


def normalize_component_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        raw_items = [str(item).strip() for item in value]
    else:
        raw_items = [str(value).strip()]
    return [item for item in raw_items if item in QWEN_QUANTIZABLE_COMPONENTS]


def quantization_info(config: Any):
    if config is None:
        return None
    if hasattr(config, "to_diff_dict"):
        return config.to_diff_dict()
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return str(config)


def build_qwen_pipeline_quantization_config(
    *,
    components: list[str],
    quantization_mode: str,
    compute_dtype: Any,
    quant_type: str,
    double_quant: bool,
):
    if quantization_mode != "bnb_4bit" or not components:
        return None

    import torch
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    from diffusers.quantizers import PipelineQuantizationConfig
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

    compute_dtype = compute_dtype or torch.bfloat16
    quant_mapping = {}
    if "transformer" in components:
        quant_mapping["transformer"] = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bool(double_quant),
        )
    if "text_encoder" in components:
        quant_mapping["text_encoder"] = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bool(double_quant),
        )
    return PipelineQuantizationConfig(quant_mapping=quant_mapping)


def diffusers_device_map_strategy(device: Any):
    device_text = str(device or "").strip().lower()
    if device_text.startswith("cuda"):
        return "cuda"
    if device_text.startswith("cpu"):
        return "cpu"
    return None


def int_in_range(value: Any, fallback: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(minimum, min(maximum, parsed))


def float_in_range(value: Any, fallback: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(minimum, min(maximum, parsed))


def parse_fill_color(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, str) or not value.strip():
        return (0, 0, 0)
    try:
        return ImageColor.getrgb(value.strip())[:3]
    except ValueError:
        raise ValueError(f"Invalid Qwen outpaint fill color: {value!r}. Use a CSS color name or hex color.")


def supports_pipeline_arg(pipeline: Any, arg_name: str) -> bool:
    try:
        return arg_name in inspect.signature(pipeline.__call__).parameters
    except (TypeError, ValueError):
        return False


def add_step_progress_callback(node: NodeBase, pipeline: Any, call_kwargs: dict[str, Any], steps: int):
    if not supports_pipeline_arg(pipeline, "callback_on_step_end"):
        return

    total_steps = max(1, int(steps or 1))

    def progress_callback(pipe, step_index, timestep, callback_kwargs):
        progress = int(((step_index + 1) / total_steps) * 100)
        node.progress(
            min(100, max(0, progress)),
            phase="denoising",
            message=f"Denoising {step_index + 1}/{total_steps}",
            current_step=step_index + 1,
            total_steps=total_steps,
        )
        return callback_kwargs

    call_kwargs["callback_on_step_end"] = progress_callback
    if supports_pipeline_arg(pipeline, "callback_on_step_end_tensor_inputs"):
        call_kwargs["callback_on_step_end_tensor_inputs"] = []


class LoadPipeline(NodeBase):
    """Load a direct Diffusers Qwen Image text-to-image pipeline."""

    label = "Load Qwen Image"
    category = "Qwen Image"
    resizable = True
    params = {
        "pipeline": {"label": "Pipeline", "display": "output", "type": "qwen_image_pipeline"},
        "model_id": {
            "label": "Model",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": QWEN_IMAGE_2512_REPO},
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
        "quantization_mode": {
            "label": "Quantization",
            "type": "string",
            "options": ["none", "bnb_4bit"],
            "default": "bnb_4bit",
        },
        "quantized_components": {
            "label": "Quantized Components",
            "type": "string",
            "display": "select",
            "options": QWEN_QUANTIZABLE_COMPONENTS,
            "fieldOptions": {"multiple": True},
            "default": QWEN_QUANTIZABLE_COMPONENTS,
        },
        "bnb_4bit_quant_type": {
            "label": "4-bit Quant Type",
            "type": "string",
            "options": ["nf4", "fp4"],
            "default": "nf4",
        },
        "bnb_4bit_compute_dtype": {
            "label": "Compute DType",
            "type": "string",
            "options": ["float32", "float16", "bfloat16"],
            "default": "bfloat16",
        },
        "bnb_4bit_use_double_quant": {"label": "Double Quant", "type": "bool", "default": True},
        "auto_offload": {"label": "Auto offload", "type": "bool", "default": True},
        "offload_mode": offload_mode_param(modes=QWEN_DIRECT_OFFLOAD_MODES),
        "low_cpu_mem_usage": {"label": "Low CPU memory", "type": "bool", "default": True},
        "resolved_artifact": {"label": "Resolved Artifact", "display": "output", "type": "string"},
    }

    def execute(self, **kwargs):
        import torch
        from diffusers import QwenImagePipeline

        model_id = repo_value(kwargs.get("model_id")) or QWEN_IMAGE_2512_REPO
        dtype = str_to_dtype(kwargs.get("dtype", "bfloat16"))
        revision = none_if_blank(kwargs.get("revision"))
        device = kwargs.get("device") or DEFAULT_DEVICE
        auto_offload = bool(kwargs.get("auto_offload", True))
        offload_mode = normalize_offload_mode(kwargs.get("offload_mode") or OFFLOAD_MODE_MODEL_CPU, auto_offload=auto_offload)
        quantization_mode = str(kwargs.get("quantization_mode") or "none")
        quantized_components = normalize_component_list(kwargs.get("quantized_components"))
        compute_dtype = str_to_dtype(kwargs.get("bnb_4bit_compute_dtype") or "bfloat16")
        quant_type = str(kwargs.get("bnb_4bit_quant_type") or "nf4")
        double_quant = bool(kwargs.get("bnb_4bit_use_double_quant", True))

        if model_id == QWEN_IMAGE_2512_PREQUANTIZED_REPO:
            # The fallback artifact is already quantized. Do not quantize it a second time.
            quantization_mode = "none"
            quantized_components = []

        quant_config = build_qwen_pipeline_quantization_config(
            components=quantized_components,
            quantization_mode=quantization_mode,
            compute_dtype=compute_dtype,
            quant_type=quant_type,
            double_quant=double_quant,
        )
        quant_mapping = getattr(quant_config, "quant_mapping", None) if quant_config else None
        self._loader_diagnostics = {
            "node_id": self.node_id,
            "loader": "QwenImage.LoadPipeline",
            "pipeline_class": "QwenImagePipeline",
            "repo_id": model_id,
            "dtype": str(dtype),
            "graph_offload_mode": kwargs.get("offload_mode"),
            "normalized_offload_mode": offload_mode,
            "auto_offload": auto_offload,
            "quantization_mode": quantization_mode,
            "quantized_components": list(quantized_components),
            "quantization": {
                name: quantization_info(config)
                for name, config in (quant_mapping or {}).items()
            },
            "resolved_artifact": model_id,
        }

        load_kwargs = {
            "torch_dtype": dtype,
            "revision": revision,
            "low_cpu_mem_usage": bool(kwargs.get("low_cpu_mem_usage", True)),
            "local_files_only": local_files_only(model_id),
        }
        if quant_config:
            load_kwargs["quantization_config"] = quant_config
            if str(device).startswith("cuda"):
                # Diffusers expects a device-map strategy here, not a torch device such as cuda:0.
                load_kwargs["device_map"] = diffusers_device_map_strategy(device)

        logger.info("Loading Qwen Image pipeline: %s", model_id)
        self.progress(-1, phase="loading", message="Loading Qwen Image pipeline")
        pipeline = QwenImagePipeline.from_pretrained(model_id, **load_kwargs)

        self.progress(-1, phase="loading", message=f"Applying {offload_mode} offload")
        offload_result = apply_pipeline_offload(
            pipeline,
            mode=offload_mode,
            device=device,
            node_id=self.node_id,
            scope="qwen-image",
        )
        self._loader_diagnostics["offload"] = {
            "mode": offload_result.mode,
            "method": offload_result.method,
            "components": offload_result.components,
            "disk_path": offload_result.disk_path,
            "detail": offload_result.detail,
        }

        self.mm_add(pipeline, priority=2)
        return {"pipeline": pipeline, "resolved_artifact": model_id}


class Generate(NodeBase):
    """Generate an image with a direct Qwen Image Diffusers pipeline."""

    label = "Qwen Image Generate"
    category = "Qwen Image"
    resizable = True
    params = {
        "pipeline": {"label": "Pipeline", "display": "input", "type": "qwen_image_pipeline"},
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "negative_prompt": {"label": "Negative Prompt", "display": "textarea", "type": "text", "default": ""},
        "width": {"label": "Width", "type": "int", "default": 1024, "min": 16, "max": 2048, "step": 16},
        "height": {"label": "Height", "type": "int", "default": 1024, "min": 16, "max": 2048, "step": 16},
        "seed": {"label": "Seed", "type": "int", "display": "random", "default": 0, "min": 0, "max": 4294967295},
        "num_inference_steps": {"label": "Steps", "display": "slider", "type": "int", "default": 50, "min": 1, "max": 100},
        "true_cfg_scale": {"label": "Guidance", "display": "slider", "type": "float", "default": 4.0, "min": 0, "max": 20, "step": 0.1},
        "max_sequence_length": {"label": "Max Sequence Length", "type": "int", "default": 512, "min": 1, "max": 2048},
        "output_type": {"label": "Output type", "type": "string", "options": ["pil", "np", "pt"], "default": "pil"},
        "images": {"label": "Images", "display": "output", "type": "image"},
        "width_out": {"label": "Width", "display": "output", "type": "int"},
        "height_out": {"label": "Height", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        import torch

        pipeline = kwargs.get("pipeline")
        if pipeline is None:
            raise ValueError("Qwen Image pipeline is required.")

        prompt = ensure_single_prompt(none_if_blank(kwargs.get("prompt")), "prompt")
        negative_prompt = ensure_single_prompt(none_if_blank(kwargs.get("negative_prompt")), "negative prompt")
        width = int(kwargs.get("width", 1024))
        height = int(kwargs.get("height", 1024))

        device = getattr(pipeline, "_execution_device", None) or getattr(pipeline, "device", None) or "cpu"
        try:
            generator = torch.Generator(device=device).manual_seed(int(kwargs.get("seed", 0)))
        except Exception:
            generator = torch.Generator(device="cpu").manual_seed(int(kwargs.get("seed", 0)))

        steps = int(kwargs.get("num_inference_steps", 50))
        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt if negative_prompt is not None else " ",
            "true_cfg_scale": float(kwargs.get("true_cfg_scale", 4.0)),
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "max_sequence_length": int(kwargs.get("max_sequence_length", 512)),
            "output_type": kwargs.get("output_type", "pil"),
            "generator": generator,
            "return_dict": True,
        }
        add_step_progress_callback(self, pipeline, call_kwargs, steps)

        result = pipeline(**call_kwargs)
        images = getattr(result, "images", result)
        return {
            "images": images,
            "width_out": width,
            "height_out": height,
        }


def placement_offset(available: int, start_margin: int, end_margin: int) -> int:
    if available <= 0:
        return 0
    requested = start_margin + end_margin
    if requested <= 0:
        return available // 2
    if requested > available:
        return int(round((start_margin / requested) * available))
    return min(start_margin, available)


def outpaint_source_size(source_size: tuple[int, int], target_size: tuple[int, int], margins: tuple[int, int, int, int]) -> tuple[int, int]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    left, right, top, bottom = margins
    available_width = max(1, target_width - left - right)
    available_height = max(1, target_height - top - bottom)
    scale = min(
        target_width / source_width,
        target_height / source_height,
        available_width / source_width,
        available_height / source_height,
        1.0,
    )
    return (
        max(1, int(round(source_width * scale))),
        max(1, int(round(source_height * scale))),
    )


class OutpaintCanvas(NodeBase):
    """Prepare an expanded canvas and boundary mask for Qwen inpaint outpainting."""

    label = "Qwen Outpaint Canvas"
    category = "Qwen Image"
    resizable = True
    params = {
        "image": {"label": "Source image", "display": "input", "type": "image"},
        "width": {"label": "Canvas width", "type": "int", "default": 1344, "min": 64, "max": 2048, "step": 16},
        "height": {"label": "Canvas height", "type": "int", "default": 768, "min": 64, "max": 2048, "step": 16},
        "left": {"label": "Left margin", "type": "int", "default": 256, "min": 0, "max": 2048, "step": 16},
        "right": {"label": "Right margin", "type": "int", "default": 256, "min": 0, "max": 2048, "step": 16},
        "top": {"label": "Top margin", "type": "int", "default": 0, "min": 0, "max": 2048, "step": 16},
        "bottom": {"label": "Bottom margin", "type": "int", "default": 0, "min": 0, "max": 2048, "step": 16},
        "overlap": {"label": "Seam overlap", "type": "int", "default": 24, "min": 0, "max": 256, "step": 4},
        "feather": {"label": "Mask feather", "type": "float", "default": 8.0, "min": 0, "max": 128, "step": 1},
        "fill_color": {"label": "Fill color", "type": "string", "default": "#000000"},
        "canvas": {"label": "Canvas", "display": "output", "type": "image"},
        "mask_image": {"label": "Mask image", "display": "output", "type": "image"},
        "width_out": {"label": "Width", "display": "output", "type": "int"},
        "height_out": {"label": "Height", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        source = ensure_single_image(kwargs.get("image"), "source image")
        target_width = int_in_range(kwargs.get("width"), 1344, 64, 2048)
        target_height = int_in_range(kwargs.get("height"), 768, 64, 2048)
        left = int_in_range(kwargs.get("left"), 256, 0, target_width)
        right = int_in_range(kwargs.get("right"), 256, 0, target_width)
        top = int_in_range(kwargs.get("top"), 0, 0, target_height)
        bottom = int_in_range(kwargs.get("bottom"), 0, 0, target_height)
        overlap = int_in_range(kwargs.get("overlap"), 24, 0, 256)
        feather = float_in_range(kwargs.get("feather"), 8.0, 0.0, 128.0)
        fill_color = parse_fill_color(kwargs.get("fill_color", "#000000"))

        source_rgba = source.convert("RGBA")
        pasted_width, pasted_height = outpaint_source_size(
            source_rgba.size,
            (target_width, target_height),
            (left, right, top, bottom),
        )
        if (pasted_width, pasted_height) != source_rgba.size:
            source_rgba = source_rgba.resize((pasted_width, pasted_height), Image.Resampling.LANCZOS)

        offset_x = placement_offset(target_width - pasted_width, left, right)
        offset_y = placement_offset(target_height - pasted_height, top, bottom)
        canvas = Image.new("RGB", (target_width, target_height), fill_color)
        canvas.paste(source_rgba, (offset_x, offset_y), source_rgba)

        mask = Image.new("L", (target_width, target_height), 255)
        keep_left = min(target_width, max(0, offset_x + overlap))
        keep_top = min(target_height, max(0, offset_y + overlap))
        keep_right = min(target_width, max(0, offset_x + pasted_width - overlap))
        keep_bottom = min(target_height, max(0, offset_y + pasted_height - overlap))
        if keep_right > keep_left and keep_bottom > keep_top:
            ImageDraw.Draw(mask).rectangle((keep_left, keep_top, keep_right, keep_bottom), fill=0)
        if feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))

        return {
            "canvas": canvas,
            "mask_image": mask,
            "width_out": target_width,
            "height_out": target_height,
        }


class LoadInpaintPipeline(NodeBase):
    """Load a direct Diffusers Qwen Image Edit inpaint pipeline."""

    label = "Load Qwen Inpaint"
    category = "Qwen Image"
    resizable = True
    params = {
        "pipeline": {"label": "Pipeline", "display": "output", "type": "qwen_image_inpaint_pipeline"},
        "model_id": {
            "label": "Model",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": QWEN_IMAGE_EDIT_DEFAULT_REPO},
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
        "quant_config": {"label": "Quantization Config", "display": "input", "type": "quant_config"},
        "auto_offload": {"label": "Auto offload", "type": "bool", "default": True},
        "offload_mode": offload_mode_param(),
        "low_cpu_mem_usage": {"label": "Low CPU memory", "type": "bool", "default": True},
    }

    def execute(self, **kwargs):
        import torch
        from diffusers import QwenImageEditInpaintPipeline

        model_id = repo_value(kwargs.get("model_id")) or QWEN_IMAGE_EDIT_DEFAULT_REPO
        dtype = str_to_dtype(kwargs.get("dtype", "bfloat16"))
        revision = none_if_blank(kwargs.get("revision"))
        device = kwargs.get("device") or DEFAULT_DEVICE
        auto_offload = bool(kwargs.get("auto_offload", True))
        offload_mode = normalize_offload_mode(kwargs.get("offload_mode") or OFFLOAD_MODE_MODEL_CPU, auto_offload=auto_offload)

        load_kwargs = {
            "torch_dtype": dtype,
            "revision": revision,
            "low_cpu_mem_usage": bool(kwargs.get("low_cpu_mem_usage", True)),
            "local_files_only": local_files_only(model_id),
        }
        quant_config = kwargs.get("quant_config")
        if quant_config:
            load_kwargs["quantization_config"] = coerce_pipeline_quantization_config(quant_config)

        logger.info("Loading Qwen Image Edit inpaint pipeline: %s", model_id)
        self.progress(-1, phase="loading", message="Loading Qwen inpaint pipeline")
        pipeline = QwenImageEditInpaintPipeline.from_pretrained(model_id, **load_kwargs)

        self.progress(-1, phase="loading", message=f"Applying {offload_mode} offload")
        apply_pipeline_offload(
            pipeline,
            mode=offload_mode,
            device=device,
            node_id=self.node_id,
            scope="qwen-image",
        )

        self.mm_add(pipeline, priority=2)
        return {"pipeline": pipeline}


class Inpaint(NodeBase):
    """Run direct Qwen Image Edit inpaint with a source image and mask image."""

    label = "Qwen Inpaint"
    category = "Qwen Image"
    resizable = True
    params = {
        "pipeline": {"label": "Pipeline", "display": "input", "type": "qwen_image_inpaint_pipeline"},
        "image": {"label": "Source image", "display": "input", "type": "image"},
        "mask_image": {"label": "Mask image", "display": "input", "type": "image"},
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "negative_prompt": {"label": "Negative Prompt", "display": "textarea", "type": "text", "default": ""},
        "width": {"label": "Width", "type": "int", "default": 1024, "min": 16, "max": 2048, "step": 16},
        "height": {"label": "Height", "type": "int", "default": 1024, "min": 16, "max": 2048, "step": 16},
        "seed": {"label": "Seed", "type": "int", "display": "random", "default": 0, "min": 0, "max": 4294967295},
        "num_inference_steps": {"label": "Steps", "display": "slider", "type": "int", "default": 40, "min": 1, "max": 100},
        "true_cfg_scale": {"label": "Guidance", "display": "slider", "type": "float", "default": 4.0, "min": 0, "max": 20, "step": 0.1},
        "strength": {"label": "Strength", "display": "slider", "type": "float", "default": 0.6, "min": 0, "max": 1, "step": 0.05},
        "padding_mask_crop": {"label": "Padding Mask Crop", "type": "int", "default": 0, "min": 0, "max": 512, "step": 8},
        "max_sequence_length": {"label": "Max Sequence Length", "type": "int", "default": 512, "min": 1, "max": 2048},
        "output_type": {"label": "Output type", "type": "string", "options": ["pil", "np", "pt"], "default": "pil"},
        "images": {"label": "Images", "display": "output", "type": "image"},
        "width_out": {"label": "Width", "display": "output", "type": "int"},
        "height_out": {"label": "Height", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        import torch

        pipeline = kwargs.get("pipeline")
        if pipeline is None:
            raise ValueError("Qwen Image inpaint pipeline is required.")

        image = ensure_single_image(kwargs.get("image"), "source image").convert("RGB")
        mask_image = ensure_single_image(kwargs.get("mask_image"), "mask image").convert("L")
        prompt = ensure_single_prompt(none_if_blank(kwargs.get("prompt")), "prompt")
        negative_prompt = ensure_single_prompt(none_if_blank(kwargs.get("negative_prompt")), "negative prompt")
        width = int(kwargs.get("width", 1024))
        height = int(kwargs.get("height", 1024))

        device = getattr(pipeline, "_execution_device", None) or "cpu"
        generator = torch.Generator(device=device).manual_seed(int(kwargs.get("seed", 0)))

        steps = int(kwargs.get("num_inference_steps", 40))
        call_kwargs = {
            "image": image,
            "mask_image": mask_image,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "true_cfg_scale": float(kwargs.get("true_cfg_scale", 4.0)),
            "height": height,
            "width": width,
            "padding_mask_crop": normalize_padding_mask_crop(kwargs.get("padding_mask_crop")),
            "strength": float(kwargs.get("strength", 0.6)),
            "num_inference_steps": steps,
            "max_sequence_length": int(kwargs.get("max_sequence_length", 512)),
            "output_type": kwargs.get("output_type", "pil"),
            "generator": generator,
            "return_dict": True,
        }
        add_step_progress_callback(self, pipeline, call_kwargs, steps)

        result = pipeline(**call_kwargs)
        images = getattr(result, "images", result)
        return {
            "images": images,
            "width_out": width,
            "height_out": height,
        }
