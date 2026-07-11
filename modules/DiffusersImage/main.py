import inspect
import logging
from typing import Any

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
from utils.huggingface import local_files_only
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST, str_to_dtype

logger = logging.getLogger("modiff")

FLUX_SCHNELL_REPO = "black-forest-labs/FLUX.1-schnell"
DEVICE_OPTIONS = list(DEVICE_LIST.keys())
IMAGE_PIPELINE_CLASSES = [
    "FluxPipeline",
    "FluxImg2ImgPipeline",
    "FluxInpaintPipeline",
    "FluxFillPipeline",
    "FluxControlPipeline",
    "FluxControlNetPipeline",
    "FluxKontextPipeline",
]
DIFFUSERS_IMAGE_OFFLOAD_MODES = [
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
]
QUANT_COMPONENTS = ["transformer", "text_encoder", "text_encoder_2", "vae"]


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


def normalize_component_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        raw_items = [str(item).strip() for item in value]
    else:
        raw_items = [str(value).strip()]
    return [item for item in raw_items if item in QUANT_COMPONENTS]


def supports_arg(pipeline: Any, arg_name: str) -> bool:
    try:
        return arg_name in inspect.signature(pipeline.__call__).parameters
    except (TypeError, ValueError):
        return False


def pipeline_class_from_name(name: str):
    import diffusers

    pipeline_class = getattr(diffusers, name, None)
    if pipeline_class is None:
        raise ValueError(f"Diffusers does not expose pipeline class {name}. Update diffusers or choose another class.")
    return pipeline_class


def quant_config_for(method: str, dtype: Any):
    if method == "bnb_4bit":
        from diffusers import BitsAndBytesConfig

        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
    if method == "bnb_8bit":
        from diffusers import BitsAndBytesConfig

        return BitsAndBytesConfig(load_in_8bit=True)
    if method == "quanto_float8":
        from diffusers import QuantoConfig

        return QuantoConfig(weights="float8")
    if method == "torchao_float8":
        from diffusers import TorchAoConfig

        return TorchAoConfig(quant_type="float8wo_e4m3")
    return None


def build_pipeline_quantization_config(method: str, components: list[str], dtype: Any):
    if method == "none" or not components:
        return None
    config = quant_config_for(method, dtype)
    if config is None:
        return None
    from diffusers.quantizers import PipelineQuantizationConfig

    return PipelineQuantizationConfig(quant_mapping={component: config for component in components})


def add_progress_callback(node: NodeBase, pipeline: Any, call_kwargs: dict[str, Any], steps: int):
    if not supports_arg(pipeline, "callback_on_step_end"):
        return

    total_steps = max(1, int(steps or 1))

    def callback(pipe, step_index, timestep, callback_kwargs):
        progress = int(((step_index + 1) / total_steps) * 100)
        node.progress(
            min(100, max(0, progress)),
            phase="denoising",
            message=f"Denoising {step_index + 1}/{total_steps}",
            current_step=step_index + 1,
            total_steps=total_steps,
        )
        return callback_kwargs

    call_kwargs["callback_on_step_end"] = callback
    if supports_arg(pipeline, "callback_on_step_end_tensor_inputs"):
        call_kwargs["callback_on_step_end_tensor_inputs"] = []


class LoadPipeline(NodeBase):
    """Load a generic Diffusers image pipeline."""

    label = "Load Diffusers Image Pipeline"
    category = "Diffusers Image"
    resizable = True
    params = {
        "pipeline": {"label": "Pipeline", "display": "output", "type": "image_diffusion_pipeline"},
        "model_id": {
            "label": "Model",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": FLUX_SCHNELL_REPO},
            "fieldOptions": {"noValidation": True, "sources": ["hub", "local"]},
        },
        "pipeline_class": {
            "label": "Pipeline Class",
            "type": "string",
            "options": IMAGE_PIPELINE_CLASSES,
            "default": "FluxPipeline",
            "fieldOptions": {"noValidation": True},
        },
        "revision": {"label": "Revision", "type": "string", "default": ""},
        "dtype": {
            "label": "DType",
            "type": "string",
            "options": ["float32", "float16", "bfloat16"],
            "default": "bfloat16",
        },
        "device": {"label": "Device", "type": "string", "options": DEVICE_OPTIONS, "default": DEFAULT_DEVICE},
        "quantization_mode": {
            "label": "Quantization",
            "type": "string",
            "options": ["none", "bnb_4bit", "bnb_8bit", "quanto_float8", "torchao_float8"],
            "default": "none",
        },
        "quantized_components": {
            "label": "Quantized Components",
            "type": "string",
            "display": "select",
            "options": QUANT_COMPONENTS,
            "fieldOptions": {"multiple": True},
            "default": [],
        },
        "auto_offload": {"label": "Auto offload", "type": "bool", "default": True},
        "offload_mode": offload_mode_param(modes=DIFFUSERS_IMAGE_OFFLOAD_MODES),
        "enable_vae_slicing": {"label": "VAE slicing", "type": "bool", "default": True},
        "enable_vae_tiling": {"label": "VAE tiling", "type": "bool", "default": True},
        "low_cpu_mem_usage": {"label": "Low CPU memory", "type": "bool", "default": True},
        "resolved_artifact": {"label": "Resolved Artifact", "display": "output", "type": "string"},
    }

    def execute(self, **kwargs):
        model_id = repo_value(kwargs.get("model_id")) or FLUX_SCHNELL_REPO
        pipeline_class_name = str(kwargs.get("pipeline_class") or "FluxPipeline")
        pipeline_class = pipeline_class_from_name(pipeline_class_name)
        dtype = str_to_dtype(kwargs.get("dtype") or "bfloat16")
        device = kwargs.get("device") or DEFAULT_DEVICE
        revision = none_if_blank(kwargs.get("revision"))
        auto_offload = bool(kwargs.get("auto_offload", True))
        offload_mode = normalize_offload_mode(kwargs.get("offload_mode") or OFFLOAD_MODE_MODEL_CPU, auto_offload=auto_offload)
        quantization_mode = str(kwargs.get("quantization_mode") or "none")
        quantized_components = normalize_component_list(kwargs.get("quantized_components"))
        quant_config = build_pipeline_quantization_config(quantization_mode, quantized_components, dtype)

        load_kwargs = {
            "torch_dtype": dtype,
            "revision": revision,
            "low_cpu_mem_usage": bool(kwargs.get("low_cpu_mem_usage", True)),
            "local_files_only": local_files_only(model_id),
        }
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
            if str(device).startswith("cuda"):
                load_kwargs["device_map"] = "cuda"

        self.progress(-1, phase="loading", message=f"Loading {pipeline_class_name}")
        pipeline = pipeline_class.from_pretrained(model_id, **load_kwargs)
        if kwargs.get("enable_vae_slicing", True) and hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        if kwargs.get("enable_vae_tiling", True) and hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()

        self.progress(-1, phase="loading", message=f"Applying {offload_mode} offload")
        apply_pipeline_offload(
            pipeline,
            mode=offload_mode,
            device=device,
            node_id=self.node_id,
            scope="diffusers-image",
        )
        self.mm_add(pipeline, priority=2)
        return {"pipeline": pipeline, "resolved_artifact": model_id}


class Generate(NodeBase):
    """Generate images from text with a Diffusers image pipeline."""

    label = "Diffusers Image Generate"
    category = "Diffusers Image"
    resizable = True
    params = {
        "pipeline": {"label": "Pipeline", "display": "input", "type": "image_diffusion_pipeline"},
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "negative_prompt": {"label": "Negative Prompt", "display": "textarea", "type": "text", "default": ""},
        "width": {"label": "Width", "type": "int", "default": 1024, "min": 16, "max": 2048, "step": 16},
        "height": {"label": "Height", "type": "int", "default": 1024, "min": 16, "max": 2048, "step": 16},
        "seed": {"label": "Seed", "type": "int", "display": "random", "default": 0, "min": 0, "max": 4294967295},
        "num_inference_steps": {"label": "Steps", "display": "slider", "type": "int", "default": 4, "min": 1, "max": 100},
        "guidance_scale": {"label": "Guidance", "display": "slider", "type": "float", "default": 0.0, "min": 0, "max": 20, "step": 0.1},
        "strength": {"label": "Strength", "display": "slider", "type": "float", "default": 0.8, "min": 0, "max": 1, "step": 0.01},
        "max_sequence_length": {"label": "Max Sequence Length", "type": "int", "default": 256, "min": 1, "max": 2048},
        "output_type": {"label": "Output type", "type": "string", "options": ["pil", "np", "pt"], "default": "pil"},
        "images": {"label": "Images", "display": "output", "type": "image"},
        "width_out": {"label": "Width", "display": "output", "type": "int"},
        "height_out": {"label": "Height", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        import torch

        pipeline = kwargs.get("pipeline")
        if pipeline is None:
            raise ValueError("Diffusers image pipeline is required.")
        device = getattr(pipeline, "_execution_device", None) or getattr(pipeline, "device", None) or "cpu"
        try:
            generator = torch.Generator(device=device).manual_seed(int(kwargs.get("seed", 0)))
        except Exception:
            generator = torch.Generator(device="cpu").manual_seed(int(kwargs.get("seed", 0)))
        steps = int(kwargs.get("num_inference_steps") or 4)
        call_kwargs = {
            "prompt": kwargs.get("prompt") or "",
            "width": int(kwargs.get("width") or 1024),
            "height": int(kwargs.get("height") or 1024),
            "num_inference_steps": steps,
            "generator": generator,
            "output_type": kwargs.get("output_type") or "pil",
            "return_dict": True,
        }
        for key in ("negative_prompt", "guidance_scale", "max_sequence_length", "strength"):
            if supports_arg(pipeline, key):
                call_kwargs[key] = kwargs.get(key)
        add_progress_callback(self, pipeline, call_kwargs, steps)
        result = pipeline(**call_kwargs)
        images = getattr(result, "images", result)
        return {"images": images, "width_out": call_kwargs["width"], "height_out": call_kwargs["height"]}


class Edit(Generate):
    """Edit an image with a Diffusers image pipeline."""

    label = "Diffusers Image Edit"
    category = "Diffusers Image"
    params = {
        **Generate.params,
        "image": {"label": "Image", "display": "input", "type": "image"},
    }

    def execute(self, **kwargs):
        if kwargs.get("image") is None:
            raise ValueError("Diffusers Image Edit needs an input image.")
        return self._execute_conditioned(kwargs, {"image": kwargs.get("image")})

    def _execute_conditioned(self, kwargs, extra_kwargs):
        import torch

        pipeline = kwargs.get("pipeline")
        if pipeline is None:
            raise ValueError("Diffusers image pipeline is required.")
        device = getattr(pipeline, "_execution_device", None) or getattr(pipeline, "device", None) or "cpu"
        try:
            generator = torch.Generator(device=device).manual_seed(int(kwargs.get("seed", 0)))
        except Exception:
            generator = torch.Generator(device="cpu").manual_seed(int(kwargs.get("seed", 0)))
        steps = int(kwargs.get("num_inference_steps") or 4)
        call_kwargs = {
            "prompt": kwargs.get("prompt") or "",
            "num_inference_steps": steps,
            "generator": generator,
            "output_type": kwargs.get("output_type") or "pil",
            "return_dict": True,
            **extra_kwargs,
        }
        for key in ("negative_prompt", "width", "height", "guidance_scale", "max_sequence_length", "strength"):
            if supports_arg(pipeline, key):
                call_kwargs[key] = kwargs.get(key)
        add_progress_callback(self, pipeline, call_kwargs, steps)
        result = pipeline(**call_kwargs)
        images = getattr(result, "images", result)
        return {"images": images, "width_out": int(kwargs.get("width") or 0), "height_out": int(kwargs.get("height") or 0)}


class Inpaint(Edit):
    """Inpaint or fill with a Diffusers image pipeline."""

    label = "Diffusers Image Inpaint"
    category = "Diffusers Image"
    params = {
        **Edit.params,
        "mask_image": {"label": "Mask", "display": "input", "type": "image"},
    }

    def execute(self, **kwargs):
        if kwargs.get("image") is None or kwargs.get("mask_image") is None:
            raise ValueError("Diffusers Image Inpaint needs image and mask_image inputs.")
        return self._execute_conditioned(kwargs, {"image": kwargs.get("image"), "mask_image": kwargs.get("mask_image")})


class ControlGenerate(Edit):
    """Generate from a control image with a Diffusers image pipeline."""

    label = "Diffusers Control Generate"
    category = "Diffusers Image"
    params = {
        **Generate.params,
        "control_image": {"label": "Control Image", "display": "input", "type": "image"},
    }

    def execute(self, **kwargs):
        if kwargs.get("control_image") is None:
            raise ValueError("Diffusers Control Generate needs a control_image input.")
        return self._execute_conditioned(kwargs, {"control_image": kwargs.get("control_image")})


class LoadAdapter(NodeBase):
    """Load a LoRA or adapter into a Diffusers image pipeline."""

    label = "Load Diffusers Image Adapter"
    category = "Diffusers Image"
    resizable = True
    params = {
        "pipeline": {"label": "Pipeline", "display": "input", "type": "image_diffusion_pipeline"},
        "adapter_path": {"label": "Adapter", "display": "modelselect", "type": "string", "fieldOptions": {"noValidation": True, "sources": ["hub", "local"]}},
        "weight_name": {"label": "Weight name", "type": "string", "default": ""},
        "adapter_name": {"label": "Adapter name", "type": "string", "default": "default"},
        "scale": {"label": "Scale", "display": "slider", "type": "float", "default": 1.0, "min": -2, "max": 2, "step": 0.01},
        "output": {"label": "Pipeline", "display": "output", "type": "image_diffusion_pipeline"},
    }

    def execute(self, **kwargs):
        pipeline = kwargs.get("pipeline")
        if pipeline is None:
            raise ValueError("LoadAdapter needs a pipeline input.")
        adapter_path = repo_value(kwargs.get("adapter_path"))
        if not adapter_path:
            return {"output": pipeline}
        if not hasattr(pipeline, "load_lora_weights"):
            raise ValueError("This pipeline does not expose load_lora_weights().")
        load_kwargs = {
            "adapter_name": kwargs.get("adapter_name") or "default",
        }
        if none_if_blank(kwargs.get("weight_name")):
            load_kwargs["weight_name"] = kwargs.get("weight_name")
        pipeline.load_lora_weights(adapter_path, **load_kwargs)
        if hasattr(pipeline, "set_adapters"):
            pipeline.set_adapters([load_kwargs["adapter_name"]], [float(kwargs.get("scale") or 1.0)])
        return {"output": pipeline}
