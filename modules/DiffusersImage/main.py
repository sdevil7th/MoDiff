import inspect
import hashlib
import logging
from dataclasses import dataclass
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
from modiff.model_artifact_catalog import require_catalog_revision, resolve_model_revision
from utils.huggingface import local_files_only
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST, str_to_dtype

logger = logging.getLogger("modiff")

FLUX_SCHNELL_REPO = "black-forest-labs/FLUX.1-schnell"
FLUX_DEV_REPO = "black-forest-labs/FLUX.1-dev"
QWEN_IMAGE_2512_REPO = "Qwen/Qwen-Image-2512"
QWEN_IMAGE_2512_PREQUANTIZED_REPO = "unsloth/Qwen-Image-2512-unsloth-bnb-4bit"
DEVICE_OPTIONS = list(DEVICE_LIST.keys())


@dataclass(frozen=True)
class ImagePipelineAdapter:
    pipeline_class: str
    modes: frozenset[str]
    default_repo: str = FLUX_SCHNELL_REPO
    guidance_parameter: str = "guidance_scale"
    multi_image_strategy: str = "list"

    def apply_generation_parameters(self, pipeline: Any, values: dict[str, Any], target: dict[str, Any]) -> None:
        aliases = {
            "negative_prompt": "negative_prompt",
            "width": "width",
            "height": "height",
            "max_sequence_length": "max_sequence_length",
            "strength": "strength",
            "padding_mask_crop": "padding_mask_crop",
            "reference_strength": "reference_strength",
        }
        for source, destination in aliases.items():
            value = values.get(source)
            # The graph uses zero to mean "no crop". Diffusers uses None for
            # that contract; passing 0 enters Qwen's overlay path and can make
            # its internally resized image conflict with the original-size
            # mask during wide outpaint finalization.
            if source == "padding_mask_crop" and (value is None or int(value) <= 0):
                continue
            if supports_arg(pipeline, destination) and value is not None:
                target[destination] = value
        if supports_arg(pipeline, self.guidance_parameter) and values.get("guidance_scale") is not None:
            target[self.guidance_parameter] = values.get("guidance_scale")


IMAGE_PIPELINE_ADAPTERS = {
    "QwenImagePipeline": ImagePipelineAdapter(
        "QwenImagePipeline",
        frozenset({"text_to_image"}),
        default_repo=QWEN_IMAGE_2512_REPO,
        guidance_parameter="true_cfg_scale",
    ),
    "ZImagePipeline": ImagePipelineAdapter("ZImagePipeline", frozenset({"text_to_image"})),
    "FluxPipeline": ImagePipelineAdapter("FluxPipeline", frozenset({"text_to_image"})),
    "Flux2KleinPipeline": ImagePipelineAdapter(
        "Flux2KleinPipeline", frozenset({"text_to_image", "edit_image", "multi_image_reference_edit"})
    ),
    "FluxImg2ImgPipeline": ImagePipelineAdapter(
        "FluxImg2ImgPipeline", frozenset({"edit_image", "multi_image_reference_edit"})
    ),
    "FluxInpaintPipeline": ImagePipelineAdapter("FluxInpaintPipeline", frozenset({"inpaint"})),
    "FluxFillPipeline": ImagePipelineAdapter("FluxFillPipeline", frozenset({"inpaint", "outpaint"})),
    "FluxControlPipeline": ImagePipelineAdapter("FluxControlPipeline", frozenset({"control_image"})),
    "FluxControlNetPipeline": ImagePipelineAdapter("FluxControlNetPipeline", frozenset({"control_image"})),
    "FluxKontextPipeline": ImagePipelineAdapter(
        "FluxKontextPipeline",
        frozenset({"edit_image", "multi_image_reference_edit"}),
        multi_image_strategy="stitch_horizontal",
    ),
    # Virtual adapter class: FLUX Redux is a prior that supplies embeddings to
    # a base FLUX pipeline, not a standalone img2img checkpoint.
    "FluxReduxPipeline": ImagePipelineAdapter(
        "FluxReduxPipeline",
        frozenset({"edit_image", "multi_image_reference_edit"}),
        # Current Diffusers performs the documented per-reference scaling and
        # weighted sum inside FluxPriorReduxPipeline. Keep the references as a
        # list and delegate the conditioning math to the upstream pipeline.
        multi_image_strategy="upstream_weighted_sum",
    ),
    "QwenImageEditInpaintPipeline": ImagePipelineAdapter(
        "QwenImageEditInpaintPipeline",
        frozenset({"inpaint", "outpaint"}),
        default_repo="Qwen/Qwen-Image-Edit",
        guidance_parameter="true_cfg_scale",
    ),
}
IMAGE_PIPELINE_CLASSES = list(IMAGE_PIPELINE_ADAPTERS)
IMAGE_PIPELINE_MODES = {name: set(adapter.modes) for name, adapter in IMAGE_PIPELINE_ADAPTERS.items()}
IMAGE_PIPELINE_MODE_OPTIONS = [
    "text_to_image",
    "edit_image",
    "multi_image_reference_edit",
    "inpaint",
    "outpaint",
    "control_image",
]
DIFFUSERS_IMAGE_OFFLOAD_MODES = [
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
]
QUANT_COMPONENTS = ["transformer", "transformer_2", "text_encoder", "text_encoder_2", "vae"]


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


def output_image_dimensions(images: Any, output_type: str = "pil") -> tuple[int | None, int | None]:
    """Return width/height for Diffusers PIL, NumPy, or Torch outputs."""

    first = images[0] if isinstance(images, (list, tuple)) and images else images
    size = getattr(first, "size", None)
    if isinstance(size, (tuple, list)) and len(size) >= 2:
        return int(size[0]), int(size[1])

    shape_value = getattr(first, "shape", None)
    if shape_value is None:
        shape_value = getattr(images, "shape", None)
    try:
        shape = tuple(int(value) for value in shape_value)
    except (TypeError, ValueError):
        return None, None
    if len(shape) < 2:
        return None, None
    if str(output_type).lower() == "pt":
        height, width = shape[-2], shape[-1]
    elif len(shape) >= 4:
        height, width = shape[-3], shape[-2]
    else:
        height, width = shape[0], shape[1]
    return width, height


def ensure_single_image(value: Any, field_name: str) -> Image.Image:
    if value in (None, ""):
        raise ValueError(f"Outpaint Canvas requires a {field_name}.")
    if isinstance(value, list):
        images = [item for item in value if item not in (None, "")]
        if len(images) != 1:
            raise ValueError(f"Outpaint Canvas requires exactly one {field_name}; received {len(images)}.")
        value = images[0]
    if not isinstance(value, Image.Image):
        raise ValueError(f"Outpaint Canvas {field_name} must be a PIL image loaded by the Image Load node.")
    return value


def _int_in_range(value: Any, fallback: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(minimum, min(maximum, parsed))


def _float_in_range(value: Any, fallback: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = fallback
    return max(minimum, min(maximum, parsed))


def _placement_offset(available: int, start_margin: int, end_margin: int) -> int:
    if available <= 0:
        return 0
    requested = start_margin + end_margin
    if requested <= 0:
        return available // 2
    if requested > available:
        return int(round((start_margin / requested) * available))
    return min(start_margin, available)


def _outpaint_source_size(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    margins: tuple[int, int, int, int],
) -> tuple[int, int]:
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
    """Prepare a model-neutral expanded canvas and white-generate boundary mask."""

    label = "Outpaint Canvas"
    category = "Diffusers Image"
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
        target_width = _int_in_range(kwargs.get("width"), 1344, 64, 2048)
        target_height = _int_in_range(kwargs.get("height"), 768, 64, 2048)
        left = _int_in_range(kwargs.get("left"), 256, 0, target_width)
        right = _int_in_range(kwargs.get("right"), 256, 0, target_width)
        top = _int_in_range(kwargs.get("top"), 0, 0, target_height)
        bottom = _int_in_range(kwargs.get("bottom"), 0, 0, target_height)
        overlap = _int_in_range(kwargs.get("overlap"), 24, 0, 256)
        feather = _float_in_range(kwargs.get("feather"), 8.0, 0.0, 128.0)
        try:
            fill_color = ImageColor.getrgb(str(kwargs.get("fill_color") or "#000000"))[:3]
        except ValueError as exc:
            raise ValueError("Outpaint Canvas fill color must be a CSS color name or hex color.") from exc

        source_rgba = source.convert("RGBA")
        pasted_width, pasted_height = _outpaint_source_size(
            source_rgba.size,
            (target_width, target_height),
            (left, right, top, bottom),
        )
        if (pasted_width, pasted_height) != source_rgba.size:
            source_rgba = source_rgba.resize((pasted_width, pasted_height), Image.Resampling.LANCZOS)

        offset_x = _placement_offset(target_width - pasted_width, left, right)
        offset_y = _placement_offset(target_height - pasted_height, top, bottom)
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


def composite_masked_pil_outputs(outputs: Any, source: Any, mask: Any) -> list[Image.Image]:
    """Keep generic inpaint outputs byte-stable outside a white-generate mask."""
    source_image = source if isinstance(source, Image.Image) else None
    mask_image = mask if isinstance(mask, Image.Image) else None
    generated = (
        [outputs]
        if isinstance(outputs, Image.Image)
        else list(outputs or [])
        if isinstance(outputs, (list, tuple))
        else []
    )
    if (
        source_image is None
        or mask_image is None
        or not generated
        or any(not isinstance(item, Image.Image) for item in generated)
    ):
        raise ValueError(
            "Diffusers image inpaint requires PIL source, mask, and output images for mask-safe compositing."
        )

    composited = []
    for image in generated:
        target_size = image.size
        normalized_source = source_image.convert("RGB")
        if normalized_source.size != target_size:
            normalized_source = normalized_source.resize(target_size, Image.Resampling.LANCZOS)
        normalized_mask = mask_image.convert("L")
        if normalized_mask.size != target_size:
            normalized_mask = normalized_mask.resize(target_size, Image.Resampling.LANCZOS)
        composited.append(Image.composite(image.convert("RGB"), normalized_source, normalized_mask))
    return composited


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


def quant_config_for(method: str, dtype: Any, modules_to_not_convert: list[str] | None = None):
    excluded = list(dict.fromkeys(modules_to_not_convert or [])) or None
    if method == "bnb_4bit":
        from diffusers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            llm_int8_skip_modules=excluded,
        )
    if method == "bnb_8bit":
        from diffusers import BitsAndBytesConfig

        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=excluded)
    if method == "quanto_float8":
        from diffusers import QuantoConfig

        return QuantoConfig(weights_dtype="float8", modules_to_not_convert=excluded)
    if method == "quanto_int8":
        from diffusers import QuantoConfig

        return QuantoConfig(weights_dtype="int8", modules_to_not_convert=excluded)
    if method == "torchao_float8":
        from diffusers import TorchAoConfig

        try:
            from torchao.quantization import Float8WeightOnlyConfig
        except ImportError as exc:
            raise RuntimeError(
                "TorchAO FP8 requires the optional torchao quantization package. "
                "Install MoDiff's quantization dependencies and retry."
            ) from exc

        return TorchAoConfig(
            quant_type=Float8WeightOnlyConfig(),
            modules_to_not_convert=excluded,
        )
    if method == "torchao_int8_weight_only":
        from diffusers import TorchAoConfig

        try:
            from torchao.quantization import Int8WeightOnlyConfig
        except ImportError as exc:
            raise RuntimeError(
                "TorchAO INT8 weight-only quantization requires the optional torchao package. "
                "Install MoDiff's quantization dependencies and retry."
            ) from exc

        return TorchAoConfig(
            quant_type=Int8WeightOnlyConfig(),
            modules_to_not_convert=excluded,
        )
    if method in {"torchao_mxfp8", "torchao_nvfp4"}:
        from diffusers import TorchAoConfig

        try:
            if method == "torchao_mxfp8":
                from torchao.prototype.mx_formats.inference_workflow import MXDynamicActivationMXWeightConfig
                from torchao.quantization.quantize_.common import KernelPreference
                import torch

                config = MXDynamicActivationMXWeightConfig(
                    activation_dtype=torch.float8_e4m3fn,
                    weight_dtype=torch.float8_e4m3fn,
                    kernel_preference=KernelPreference.AUTO,
                )
            else:
                from torchao.prototype.mx_formats.inference_workflow import (
                    NVFP4DynamicActivationNVFP4WeightConfig,
                )

                config = NVFP4DynamicActivationNVFP4WeightConfig(
                    use_dynamic_per_tensor_scale=True,
                    use_triton_kernel=True,
                )
        except ImportError as exc:
            raise RuntimeError(
                f"{method.removeprefix('torchao_').upper()} artifact creation requires a compatible "
                "Blackwell PyTorch, TorchAO, and kernel environment."
            ) from exc
        return TorchAoConfig(quant_type=config, modules_to_not_convert=excluded)
    return None


def build_pipeline_quantization_config(method: str, components: list[str], dtype: Any):
    if method == "none" or not components:
        return None
    config = quant_config_for(method, dtype)
    if config is None:
        return None
    from diffusers.quantizers import PipelineQuantizationConfig

    return PipelineQuantizationConfig(quant_mapping={component: config for component in components})


def coerce_pipeline_quantization_config(quant_config: Any):
    """Normalize graph-provided component mappings to Diffusers' public config."""

    if not quant_config:
        return None
    if isinstance(quant_config, str):
        if not quant_config.strip():
            return None
        raise ValueError("Quantization config must be connected to a Quantization Config node output.")
    from diffusers.quantizers import PipelineQuantizationConfig

    if isinstance(quant_config, PipelineQuantizationConfig):
        return quant_config
    if isinstance(quant_config, dict):
        return PipelineQuantizationConfig(quant_mapping=quant_config)
    raise ValueError("Quantization config must be a Diffusers PipelineQuantizationConfig or component mapping.")


def build_qwen_pipeline_quantization_config(
    *,
    components: list[str],
    quantization_mode: str,
    compute_dtype: Any,
    quant_type: str = "nf4",
    double_quant: bool = True,
):
    """Build component-correct BnB configs for Qwen's Diffusers and Transformers parts."""

    if quantization_mode != "bnb_4bit" or not components:
        return None
    from importlib.util import find_spec

    if find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "BitsAndBytes 4-bit quantization is not installed. Install the CUDA quantization extra or choose another mode."
        )
    from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    from diffusers.quantizers import PipelineQuantizationConfig
    from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

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
    return PipelineQuantizationConfig(quant_mapping=quant_mapping) if quant_mapping else None


def add_progress_callback(node: NodeBase, pipeline: Any, call_kwargs: dict[str, Any], steps: int):
    node.progress(
        0,
        phase="denoising",
        message=f"Denoising 0/{steps}",
        current_step=0,
        total_steps=steps,
        elapsed_seconds=0.0,
    )
    if not supports_arg(pipeline, "callback_on_step_end"):
        return

    def callback(pipe, step_index, timestep, callback_kwargs):
        # NodeBase owns the common cancellation contract. Propagating it here
        # makes /stop interrupt generic Diffusers pipelines at the next step,
        # just like the established video/audio nodes.
        return node.pipe_callback(pipe, step_index, timestep, callback_kwargs) or callback_kwargs

    call_kwargs["callback_on_step_end"] = callback
    if supports_arg(pipeline, "callback_on_step_end_tensor_inputs"):
        call_kwargs["callback_on_step_end_tensor_inputs"] = []


def prepare_reference_images(image: Any, adapter: ImagePipelineAdapter) -> Any:
    """Translate the generic multi-reference contract for a pipeline adapter."""
    if not isinstance(image, (list, tuple)) or len(image) <= 1:
        return image[0] if isinstance(image, (list, tuple)) and image else image
    if adapter.multi_image_strategy != "stitch_horizontal":
        return image if isinstance(image, list) else list(image)

    from PIL import Image

    if not all(isinstance(item, Image.Image) for item in image):
        raise ValueError("Horizontal multi-reference stitching currently requires PIL image inputs.")
    converted = [item.convert("RGB") for item in image]
    target_height = max(item.height for item in converted)
    resized = [
        item
        if item.height == target_height
        else item.resize(
            (max(1, round(item.width * target_height / item.height)), target_height), Image.Resampling.LANCZOS
        )
        for item in converted
    ]
    canvas = Image.new("RGB", (sum(item.width for item in resized), target_height))
    left = 0
    for item in resized:
        canvas.paste(item, (left, 0))
        left += item.width
    return canvas


class FluxReduxPipelineBundle:
    """Callable adapter combining the official Redux prior with FLUX.1-dev."""

    def __init__(self, prior: Any, base: Any):
        self.prior = prior
        self.base = base
        self._modiff_image_adapter = IMAGE_PIPELINE_ADAPTERS["FluxReduxPipeline"]

    @property
    def device(self):
        return getattr(self.base, "device", "cpu")

    @property
    def _execution_device(self):
        return getattr(self.base, "_execution_device", self.device)

    def __call__(
        self,
        *,
        image,
        prompt="",
        negative_prompt=None,
        width=None,
        height=None,
        num_inference_steps=28,
        guidance_scale=3.5,
        strength=None,
        padding_mask_crop=None,
        max_sequence_length=512,
        generator=None,
        output_type="pil",
        return_dict=True,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=None,
        reference_strength=1.0,
    ):
        # FluxPriorReduxPipeline produces the exact reference/text embeddings
        # consumed by FluxPipeline. For multiple images, current Diffusers
        # applies the per-image scales and returns one upstream weighted sum.
        prior_kwargs = {"image": image, "prompt": prompt or None, "return_dict": True}
        if isinstance(image, (list, tuple)) and len(image) > 1:
            secondary_strength = float(reference_strength)
            if not 0.0 <= secondary_strength <= 1.0:
                raise ValueError("reference_strength must be between 0 and 1.")
            reference_scales = [1.0, *([secondary_strength] * (len(image) - 1))]
            prior_kwargs["prompt_embeds_scale"] = reference_scales
            prior_kwargs["pooled_prompt_embeds_scale"] = reference_scales
        prior_output = self.prior(**prior_kwargs)
        prompt_embeds = prior_output.prompt_embeds
        pooled_prompt_embeds = prior_output.pooled_prompt_embeds
        base_kwargs = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "output_type": output_type,
            "return_dict": return_dict,
            "max_sequence_length": max_sequence_length,
        }
        for key, value in (("width", width), ("height", height)):
            if value is not None:
                base_kwargs[key] = value
        if callback_on_step_end is not None:
            base_kwargs["callback_on_step_end"] = callback_on_step_end
        if callback_on_step_end_tensor_inputs is not None:
            base_kwargs["callback_on_step_end_tensor_inputs"] = callback_on_step_end_tensor_inputs
        return self.base(**base_kwargs)


class LoadPipeline(NodeBase):
    """Load a generic Diffusers image pipeline."""

    label = "Load Diffusers Image Pipeline"
    category = "Diffusers Image"
    resizable = True
    # Flux2KleinPipeline uses one resident pipeline for generation and edit
    # calls. Mode validates the requested operation but does not participate in
    # from_pretrained(), so changing it must not force another 13-minute load.
    cache_ignored_params = frozenset({"mode"})
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
        "mode": {
            "label": "Mode",
            "type": "string",
            "options": IMAGE_PIPELINE_MODE_OPTIONS,
            "default": "text_to_image",
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
            "options": [
                "none",
                "bnb_4bit",
                "bnb_8bit",
                "quanto_float8",
                "quanto_int8",
                "torchao_float8",
                "torchao_int8_weight_only",
            ],
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
        "execution_recipe": {
            "label": "Execution Recipe",
            "display": "input",
            "type": "diffusers_execution_recipe",
        },
        "device_map": {
            "label": "Device Map",
            "type": "string",
            "options": ["none", "cuda", "auto", "balanced", "balanced_low_0", "cpu"],
            "default": "none",
        },
        "auto_offload": {"label": "Auto offload", "type": "bool", "default": True},
        "offload_mode": offload_mode_param(modes=DIFFUSERS_IMAGE_OFFLOAD_MODES),
        "enable_vae_slicing": {"label": "VAE slicing", "type": "bool", "default": True},
        "enable_vae_tiling": {"label": "VAE tiling", "type": "bool", "default": True},
        "low_cpu_mem_usage": {"label": "Low CPU memory", "type": "bool", "default": True},
        "resolved_artifact": {"label": "Resolved Artifact", "display": "output", "type": "string"},
    }

    @staticmethod
    def _validate_mode(pipeline_class_name: str, requested_mode: str):
        adapter = IMAGE_PIPELINE_ADAPTERS.get(pipeline_class_name)
        if adapter is None or requested_mode not in adapter.modes:
            supported = ", ".join(sorted(adapter.modes if adapter else [])) or "none"
            raise ValueError(f"{pipeline_class_name} does not support {requested_mode}. Supported modes: {supported}.")
        return adapter

    def __call__(self, **kwargs):
        pipeline_class_name = str(kwargs.get("pipeline_class") or "FluxPipeline")
        requested_mode = str(kwargs.get("mode") or "text_to_image")
        self._validate_mode(pipeline_class_name, requested_mode)
        return super().__call__(**kwargs)

    def prepare_for_workflow_reuse(self):
        """Restore a resident image pipeline before another graph adopts it."""
        pipeline = self.output.get("pipeline")
        unload_lora_weights = getattr(pipeline, "unload_lora_weights", None)
        if callable(unload_lora_weights):
            unload_lora_weights()

    def execute(self, **kwargs):
        from modules.DiffusersRuntime.main import (
            apply_execution_recipe_to_pipeline,
            assert_runtime_quantization_full_residency,
            enable_parallel_weight_loading,
        )

        execution_recipe = kwargs.get("execution_recipe")
        if execution_recipe is None:
            execution_recipe = {}
        if not isinstance(execution_recipe, dict):
            raise TypeError("Execution Recipe must come from a Diffusers Execution Recipe node.")
        pipeline_class_name = str(kwargs.get("pipeline_class") or "FluxPipeline")
        requested_mode = str(kwargs.get("mode") or "text_to_image")
        adapter = self._validate_mode(pipeline_class_name, requested_mode)
        model_selection = kwargs.get("model_id")
        selected_model_id = repo_value(model_selection)
        model_id = selected_model_id or adapter.default_repo
        model_source = model_selection.get("source") if selected_model_id and isinstance(model_selection, dict) else "hub"
        dtype = str_to_dtype(kwargs.get("dtype") or "bfloat16")
        device = execution_recipe.get("device") or kwargs.get("device") or DEFAULT_DEVICE
        revision = resolve_model_revision(
            model_id,
            none_if_blank(kwargs.get("revision")),
            source=model_source,
        )
        auto_offload = bool(kwargs.get("auto_offload", True))
        recipe_offload = execution_recipe.get("offload_mode")
        if recipe_offload is not None:
            auto_offload = str(recipe_offload) != "none"
        offload_mode = normalize_offload_mode(
            recipe_offload if recipe_offload is not None else kwargs.get("offload_mode") or OFFLOAD_MODE_MODEL_CPU,
            auto_offload=auto_offload,
            device=device,
        )
        quantization_mode = str(kwargs.get("quantization_mode") or "none")
        quantized_components = normalize_component_list(kwargs.get("quantized_components"))
        if model_id == QWEN_IMAGE_2512_PREQUANTIZED_REPO:
            quantization_mode = "none"
            quantized_components = []
        quant_config = coerce_pipeline_quantization_config(execution_recipe.get("quantization_config"))
        if quant_config is None:
            if pipeline_class_name.startswith("QwenImage") and quantization_mode == "bnb_4bit":
                quant_config = build_qwen_pipeline_quantization_config(
                    components=quantized_components,
                    quantization_mode=quantization_mode,
                    compute_dtype=dtype,
                )
            else:
                quant_config = build_pipeline_quantization_config(quantization_mode, quantized_components, dtype)
        if quant_config is not None:
            assert_runtime_quantization_full_residency(
                model_id=model_id,
                revision=revision,
                quantization_config=quant_config,
                device=str(device),
                offload_mode=offload_mode,
                device_map=execution_recipe.get("device_map") or "none",
            )

        load_kwargs = {
            "torch_dtype": dtype,
            "revision": revision,
            "low_cpu_mem_usage": bool(kwargs.get("low_cpu_mem_usage", True)),
            "local_files_only": local_files_only(model_id),
        }
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
        recipe_device_map = execution_recipe.get("device_map")
        direct_device_map = kwargs.get("device_map")
        device_map = (
            direct_device_map
            if direct_device_map not in (None, "", "none")
            else recipe_device_map or direct_device_map or "none"
        )
        if device_map == "none" and offload_mode == "none" and str(device) in {"cuda", "cuda:0"}:
            # A no-offload complete pipeline is meant to be fully resident.
            # Stream its shards directly to the target accelerator instead of
            # materializing a second full CPU copy before pipeline.to(device).
            device_map = "cuda"
        if device_map == "cuda" and offload_mode == "none" and str(device) in {"cuda", "cuda:0"}:
            enable_parallel_weight_loading()
        if device_map != "none":
            load_kwargs["device_map"] = device_map
        elif quant_config is not None and str(device).startswith("cuda"):
            load_kwargs["device_map"] = "cuda"
        max_memory = execution_recipe.get("max_memory")
        if isinstance(max_memory, dict) and max_memory:
            load_kwargs["max_memory"] = max_memory

        self.progress(-1, phase="loading", message=f"Loading {pipeline_class_name}")
        if pipeline_class_name == "FluxReduxPipeline":
            from diffusers import FluxPipeline, FluxPriorReduxPipeline

            base_kwargs = dict(load_kwargs)
            # The graph revision belongs to the Redux prior. Its fixed FLUX.1
            # base is a separate Hub snapshot and must carry its own pin.
            base_kwargs["revision"] = require_catalog_revision(FLUX_DEV_REPO, model_type="FluxDevPipeline")
            with self.diffusers_loading_progress():
                base = FluxPipeline.from_pretrained(FLUX_DEV_REPO, **base_kwargs)

                prior_kwargs = dict(load_kwargs)
                prior_kwargs.pop("quantization_config", None)
                prior_kwargs.pop("device_map", None)
                # Redux exposes optional text components. Share the base FLUX
                # encoders/tokenizers so its documented prompt input is real rather
                # than silently ignored, then leave the base pipeline embedding-only.
                for component in ("text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"):
                    prior_kwargs[component] = getattr(base, component, None)
                prior = FluxPriorReduxPipeline.from_pretrained(model_id, **prior_kwargs)
            if hasattr(base, "register_modules"):
                base.register_modules(
                    text_encoder=None,
                    text_encoder_2=None,
                    tokenizer=None,
                    tokenizer_2=None,
                )
            pipeline = FluxReduxPipelineBundle(prior, base)
        else:
            pipeline_class = pipeline_class_from_name(pipeline_class_name)
            with self.diffusers_loading_progress():
                pipeline = pipeline_class.from_pretrained(model_id, **load_kwargs)
        pipeline._modiff_image_adapter = adapter
        runtime_recipe = {
            **execution_recipe,
            "vae_slicing": bool(
                execution_recipe.get("vae_slicing", kwargs.get("enable_vae_slicing", True))
            ),
            "vae_tiling": bool(
                execution_recipe.get("vae_tiling", kwargs.get("enable_vae_tiling", True))
            ),
        }
        runtime_owner = pipeline.base if isinstance(pipeline, FluxReduxPipelineBundle) else pipeline
        pipeline._modiff_runtime_config = apply_execution_recipe_to_pipeline(runtime_owner, runtime_recipe)

        self.progress(99, phase="component_placement", message=f"Applying {offload_mode} offload")
        offload_targets = (
            [pipeline.prior, pipeline.base] if isinstance(pipeline, FluxReduxPipelineBundle) else [pipeline]
        )
        for index, target in enumerate(offload_targets):
            offload_result = apply_pipeline_offload(
                target,
                mode=offload_mode,
                device=device,
                node_id=self.node_id,
                scope=f"diffusers-image-{index}" if len(offload_targets) > 1 else "diffusers-image",
            )
            self.progress(
                -1,
                phase="component_placement",
                message=f"Registering pipeline memory ({getattr(offload_result, 'method', 'configured')})",
            )
            self.mm_add(target, priority=2)
        return {"pipeline": pipeline, "resolved_artifact": model_id}


class Generate(NodeBase):
    """Generate images from text with a Diffusers image pipeline."""

    label = "Diffusers Image Generate"
    category = "Diffusers Image"
    resizable = True
    params = {
        "pipeline": {"label": "Pipeline", "display": "input", "type": "image_diffusion_pipeline", "required": True},
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "negative_prompt": {"label": "Negative Prompt", "display": "textarea", "type": "text", "default": ""},
        "width": {"label": "Width", "type": "int", "default": 1024, "min": 16, "max": 2048, "step": 16},
        "height": {"label": "Height", "type": "int", "default": 1024, "min": 16, "max": 2048, "step": 16},
        "seed": {"label": "Seed", "type": "int", "display": "random", "default": 0, "min": 0, "max": 4294967295},
        "num_inference_steps": {
            "label": "Steps",
            "display": "slider",
            "type": "int",
            "default": 4,
            "min": 1,
            "max": 100,
        },
        "guidance_scale": {
            "label": "Guidance",
            "display": "slider",
            "type": "float",
            "default": 0.0,
            "min": 0,
            "max": 20,
            "step": 0.1,
        },
        "strength": {
            "label": "Strength",
            "display": "slider",
            "type": "float",
            "default": 0.8,
            "min": 0,
            "max": 1,
            "step": 0.01,
        },
        "padding_mask_crop": {
            "label": "Padding Mask Crop",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 512,
            "step": 8,
        },
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
        adapter = getattr(pipeline, "_modiff_image_adapter", None) or ImagePipelineAdapter(
            type(pipeline).__name__, frozenset()
        )
        adapter.apply_generation_parameters(pipeline, kwargs, call_kwargs)
        add_progress_callback(self, pipeline, call_kwargs, steps)
        self._active_pipeline = pipeline
        try:
            result = pipeline(**call_kwargs)
        finally:
            self._active_pipeline = None
        images = getattr(result, "images", result)
        return {"images": images, "width_out": call_kwargs["width"], "height_out": call_kwargs["height"]}


class Edit(Generate):
    """Edit an image with a Diffusers image pipeline."""

    label = "Diffusers Image Edit"
    category = "Diffusers Image"
    params = {
        **Generate.params,
        "image": {
            "label": "Image or references",
            "display": "input",
            "type": "image",
            "required": True,
            "description": "A single source image or a list of references for pipelines that support multi-reference editing.",
        },
        "reference_strength": {
            "label": "Secondary reference strength",
            "type": "float",
            "default": 1.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "description": "Relative influence of every reference after the first composition anchor, when supported by the selected adapter.",
        },
    }

    def execute(self, **kwargs):
        if kwargs.get("image") is None:
            raise ValueError("Diffusers Image Edit needs an input image.")
        pipeline = kwargs.get("pipeline")
        adapter = getattr(pipeline, "_modiff_image_adapter", None) or ImagePipelineAdapter(
            type(pipeline).__name__ if pipeline is not None else "unknown", frozenset()
        )
        image = prepare_reference_images(kwargs.get("image"), adapter)
        return self._execute_conditioned(kwargs, {"image": image})

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
        adapter = getattr(pipeline, "_modiff_image_adapter", None)
        if adapter is None:
            guidance_parameter = "true_cfg_scale" if supports_arg(pipeline, "true_cfg_scale") else "guidance_scale"
            adapter = ImagePipelineAdapter(type(pipeline).__name__, frozenset(), guidance_parameter=guidance_parameter)
        adapter.apply_generation_parameters(pipeline, kwargs, call_kwargs)
        add_progress_callback(self, pipeline, call_kwargs, steps)
        self._active_pipeline = pipeline
        try:
            result = pipeline(**call_kwargs)
        finally:
            self._active_pipeline = None
        images = getattr(result, "images", result)
        actual_width, actual_height = output_image_dimensions(images, call_kwargs["output_type"])
        return {
            "images": images,
            "width_out": actual_width if actual_width is not None else int(kwargs.get("width") or 0),
            "height_out": actual_height if actual_height is not None else int(kwargs.get("height") or 0),
        }


class Inpaint(Edit):
    """Inpaint or fill with a Diffusers image pipeline."""

    label = "Diffusers Image Inpaint"
    category = "Diffusers Image"
    params = {
        **Edit.params,
        "mask_image": {"label": "Mask", "display": "input", "type": "image", "required": True},
        "output_type": {"label": "Output type", "type": "string", "options": ["pil"], "default": "pil"},
    }

    def execute(self, **kwargs):
        if kwargs.get("image") is None or kwargs.get("mask_image") is None:
            raise ValueError("Diffusers Image Inpaint needs image and mask_image inputs.")
        if kwargs.get("output_type", "pil") != "pil":
            raise ValueError("Diffusers Image Inpaint requires output_type='pil' for mask-safe compositing.")
        result = self._execute_conditioned(
            kwargs,
            {"image": kwargs.get("image"), "mask_image": kwargs.get("mask_image")},
        )
        result["images"] = composite_masked_pil_outputs(
            result.get("images"),
            kwargs.get("image"),
            kwargs.get("mask_image"),
        )
        return result


class ControlGenerate(Edit):
    """Generate from a control image with a Diffusers image pipeline."""

    label = "Diffusers Control Generate"
    category = "Diffusers Image"
    params = {
        **Generate.params,
        "control_image": {"label": "Control Image", "display": "input", "type": "image", "required": True},
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
        "pipeline": {"label": "Pipeline", "display": "input", "type": "image_diffusion_pipeline", "required": True},
        "adapter_path": {
            "label": "Adapter",
            "display": "modelselect",
            "type": "string",
            "fieldOptions": {"noValidation": True, "sources": ["hub", "local"]},
        },
        "weight_name": {"label": "Weight name", "type": "string", "default": ""},
        "expected_sha256": {
            "label": "Expected SHA-256",
            "type": "string",
            "default": "",
            "description": "Optional immutable hash for the selected adapter weight file.",
        },
        "adapter_name": {"label": "Adapter name", "type": "string", "default": "default"},
        "replace_existing": {
            "label": "Replace existing adapters",
            "type": "boolean",
            "default": True,
            "description": "Unload adapters already attached to this pipeline before loading this graph's adapter.",
        },
        "scale": {
            "label": "Scale",
            "display": "slider",
            "type": "float",
            "default": 1.0,
            "min": -2,
            "max": 2,
            "step": 0.01,
        },
        "output": {"label": "Pipeline", "display": "output", "type": "image_diffusion_pipeline"},
    }

    def execute(self, **kwargs):
        pipeline = kwargs.get("pipeline")
        if pipeline is None:
            raise ValueError("LoadAdapter needs a pipeline input.")
        adapter_selection = kwargs.get("adapter_path")
        adapter_path = repo_value(adapter_selection)
        if not adapter_path:
            return {"output": pipeline}
        if not hasattr(pipeline, "load_lora_weights"):
            raise ValueError("This pipeline does not expose load_lora_weights().")
        load_kwargs = {
            "adapter_name": kwargs.get("adapter_name") or "default",
        }
        weight_name = none_if_blank(kwargs.get("weight_name"))
        source = adapter_selection.get("source") if isinstance(adapter_selection, dict) else "hub"
        if source == "hub":
            from pathlib import Path
            from utils.huggingface import cached_file_path

            repo_id = adapter_path
            if not weight_name:
                parts = adapter_path.split("/")
                if len(parts) >= 3:
                    repo_id, weight_name = "/".join(parts[:2]), "/".join(parts[2:])
            if not weight_name:
                raise ValueError("A Hub adapter requires a pinned weight_name for app-managed installation.")
            cached = cached_file_path(repo_id, weight_name)
            if not cached:
                raise FileNotFoundError(
                    f"Adapter {repo_id}/{weight_name} is not installed. Install the pinned file through Model Manager first."
                )
            cached_path = Path(cached)
            adapter_path = str(cached_path.parent)
            weight_name = cached_path.name
            expected_sha256 = str(kwargs.get("expected_sha256") or "").strip().lower().removeprefix("sha256:")
            if expected_sha256:
                digest = hashlib.sha256()
                with cached_path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected_sha256:
                    raise ValueError(
                        f"Adapter {repo_id}/{weight_name} failed its pinned SHA-256 verification. "
                        "Repair the adapter through Model Manager before running this graph."
                    )
        if weight_name:
            load_kwargs["weight_name"] = weight_name
        replace_existing = bool(kwargs.get("replace_existing", True))
        adapter_scales = dict(getattr(pipeline, "_modiff_adapter_scales", {}) or {})
        if replace_existing and hasattr(pipeline, "unload_lora_weights"):
            pipeline.unload_lora_weights()
            adapter_scales.clear()
        pipeline.load_lora_weights(adapter_path, **load_kwargs)
        raw_scale = kwargs.get("scale")
        scale = 1.0 if raw_scale is None else float(raw_scale)
        adapter_scales[load_kwargs["adapter_name"]] = scale
        if hasattr(pipeline, "set_adapters"):
            pipeline.set_adapters(list(adapter_scales), list(adapter_scales.values()))
        pipeline._modiff_adapter_scales = adapter_scales
        return {"output": pipeline}
