"""Generic Diffusers 3D generation with a rendered-orbit output boundary."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from modiff.NodeBase import NodeBase
from modiff.diffusers_offload import (
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
    apply_pipeline_offload,
    offload_mode_param,
)
from modiff.model_artifact_catalog import require_catalog_revision
from utils.huggingface import local_files_only
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST, str_to_dtype


SHAP_E_REPO = "openai/shap-e"
SHAP_E_REVISION = "7bd337afdea1c17842e1c3cc45c4e268356dba40"
SHAP_E_PIPELINE_CLASS = "ShapEPipeline"
SHAP_E_MODE = "text_to_3d"
SHAP_E_IMG2IMG_REPO = "openai/shap-e-img2img"
SHAP_E_IMG2IMG_REVISION = "0e0aba80f08d368aaf6af9cb93583707481cc29b"
SHAP_E_IMG2IMG_PIPELINE_CLASS = "ShapEImg2ImgPipeline"
SHAP_E_IMG2IMG_MODE = "image_to_3d"
SHAP_E_RENDERER_SAFE_SUBFOLDER = "renderer"
SHAP_E_VARIANT = "fp16"
DEVICE_OPTIONS = list(DEVICE_LIST.keys())
DIRECT_THREE_D_OFFLOAD_MODES = [
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
]


@dataclass(frozen=True)
class ThreeDPipelineAdapter:
    pipeline_class: str
    default_repo: str
    revision: str
    mode: str
    default_steps: int
    default_guidance: float
    default_frame_size: int
    max_steps: int = 128
    max_frame_size: int = 256
    input_kind: str = "text"

    def signal_value(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "library": "diffusers",
            "mediaKind": "three_d",
            "pipelineClass": self.pipeline_class,
            "mode": self.mode,
            "repository": self.default_repo,
            "outputContract": "rendered_orbit",
            "inputContract": (
                {"requiredImages": ["referenceImages"]}
                if self.input_kind == "image"
                else {"requiredImages": []}
            ),
            "fieldParams": {
                "prompt": {
                    "hidden": self.input_kind != "text",
                    "required": self.input_kind == "text",
                },
                "reference_images": {
                    "hidden": self.input_kind != "image",
                    "required": self.input_kind == "image",
                },
                "num_inference_steps": {
                    "default": self.default_steps,
                    "min": 1,
                    "max": self.max_steps,
                },
                "guidance_scale": {
                    "default": self.default_guidance,
                    "min": 0,
                    "max": 20,
                    "step": 0.1,
                },
                "frame_size": {
                    "default": self.default_frame_size,
                    "min": 64,
                    "max": self.max_frame_size,
                    "step": 8,
                },
            },
        }


SHAP_E_ADAPTER = ThreeDPipelineAdapter(
    pipeline_class=SHAP_E_PIPELINE_CLASS,
    default_repo=SHAP_E_REPO,
    revision=SHAP_E_REVISION,
    mode=SHAP_E_MODE,
    default_steps=64,
    default_guidance=15.0,
    default_frame_size=256,
)
SHAP_E_IMG2IMG_ADAPTER = ThreeDPipelineAdapter(
    pipeline_class=SHAP_E_IMG2IMG_PIPELINE_CLASS,
    default_repo=SHAP_E_IMG2IMG_REPO,
    revision=SHAP_E_IMG2IMG_REVISION,
    mode=SHAP_E_IMG2IMG_MODE,
    default_steps=64,
    default_guidance=3.0,
    default_frame_size=256,
    input_kind="image",
)
THREE_D_PIPELINE_ADAPTERS = {
    SHAP_E_PIPELINE_CLASS: SHAP_E_ADAPTER,
    SHAP_E_IMG2IMG_PIPELINE_CLASS: SHAP_E_IMG2IMG_ADAPTER,
}
THREE_D_PIPELINE_CLASS_OPTIONS = [SHAP_E_IMG2IMG_PIPELINE_CLASS, SHAP_E_PIPELINE_CLASS]
THREE_D_MODE_OPTIONS = [SHAP_E_IMG2IMG_MODE, SHAP_E_MODE]
DEFAULT_THREE_D_CONTRACT = SHAP_E_ADAPTER.signal_value()


def _require_adapter(pipeline_class: Any, mode: Any) -> ThreeDPipelineAdapter:
    adapter = THREE_D_PIPELINE_ADAPTERS.get(pipeline_class)
    if adapter is None or mode != adapter.mode:
        raise ValueError("The rendered-3D loader requires one reviewed pipeline and task-mode pair.")
    return adapter


def _require_exact_selection(
    model_id: Any,
    revision: Any,
    *,
    adapter: ThreeDPipelineAdapter = SHAP_E_ADAPTER,
) -> tuple[str, str]:
    if isinstance(model_id, dict):
        if set(model_id) != {"source", "value"} or model_id.get("source") != "hub":
            raise ValueError("The reviewed Shap-E artifact must use the exact Hub selection object.")
        repository = model_id.get("value")
    else:
        repository = model_id
    if repository != adapter.default_repo:
        raise ValueError(f"The reviewed rendered-3D contract requires {adapter.default_repo}.")
    catalog_revision = require_catalog_revision(adapter.default_repo, model_type=adapter.pipeline_class)
    selected_revision = revision or catalog_revision
    if selected_revision != catalog_revision or selected_revision != adapter.revision:
        raise ValueError(f"The reviewed Shap-E artifact is pinned to {adapter.revision}.")
    return repository, selected_revision


def _bounded_int(value: Any, *, default: int, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer from {minimum} through {maximum}.")
    try:
        number = float(default if value is None else value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be an integer from {minimum} through {maximum}.") from error
    if not isfinite(number) or not number.is_integer() or not minimum <= number <= maximum:
        raise ValueError(f"{label} must be an integer from {minimum} through {maximum}.")
    return int(number)


def _bounded_float(value: Any, *, default: float, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number from {minimum:g} through {maximum:g}.")
    try:
        number = float(default if value is None else value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite number from {minimum:g} through {maximum:g}.") from error
    if not isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{label} must be a finite number from {minimum:g} through {maximum:g}.")
    return number


def _tag_pipeline(
    pipeline: Any,
    adapter: ThreeDPipelineAdapter,
    repository: str,
    revision: str,
) -> None:
    setattr(pipeline, "_modiff_three_d_pipeline_class", adapter.pipeline_class)
    setattr(pipeline, "_modiff_three_d_mode", adapter.mode)
    setattr(pipeline, "_modiff_three_d_repo", repository)
    setattr(pipeline, "_modiff_three_d_revision", revision)


def _validate_connected_pipeline(pipeline: Any, contract: Any) -> ThreeDPipelineAdapter:
    if pipeline is None:
        raise ValueError("Rendered 3D generation requires a connected Diffusers pipeline.")
    adapter = _require_adapter(
        getattr(pipeline, "_modiff_three_d_pipeline_class", None),
        getattr(pipeline, "_modiff_three_d_mode", None),
    )
    expected_contract = adapter.signal_value()
    if contract != expected_contract:
        raise ValueError("The connected 3D pipeline published a stale or mismatched task contract.")
    identity = (
        getattr(pipeline, "_modiff_three_d_pipeline_class", None),
        getattr(pipeline, "_modiff_three_d_mode", None),
        getattr(pipeline, "_modiff_three_d_repo", None),
        getattr(pipeline, "_modiff_three_d_revision", None),
    )
    expected = (adapter.pipeline_class, adapter.mode, adapter.default_repo, adapter.revision)
    if identity != expected:
        raise ValueError("The connected pipeline does not match the reviewed rendered-3D artifact identity.")
    return adapter


def _load_safe_shap_e_components(
    repository: str,
    revision: str,
    *,
    dtype: Any,
    low_cpu_mem_usage: bool,
    adapter: ThreeDPipelineAdapter = SHAP_E_ADAPTER,
) -> dict[str, Any]:
    """Load only the reviewed fp16 safetensors component envelope.

    The current repository retains a legacy pickle only in
    ``shap_e_renderer/``. Its earlier ``renderer/`` folder contains the same
    official renderer in safetensors form, so explicit assembly is required to
    prevent ``DiffusionPipeline.from_pretrained`` from falling back to pickle.
    """

    from diffusers import HeunDiscreteScheduler, PriorTransformer
    from diffusers.pipelines.shap_e.renderer import ShapERenderer
    from transformers import (
        CLIPImageProcessor,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
        CLIPVisionModel,
    )

    common = {
        "revision": revision,
        "local_files_only": local_files_only(repository),
    }
    model_common = {
        **common,
        "torch_dtype": dtype,
        "variant": SHAP_E_VARIANT,
        "use_safetensors": True,
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }
    components = {
        "prior": PriorTransformer.from_pretrained(repository, subfolder="prior", **model_common),
        "scheduler": HeunDiscreteScheduler.from_pretrained(repository, subfolder="scheduler", **common),
        "shap_e_renderer": ShapERenderer.from_pretrained(
            repository,
            subfolder=SHAP_E_RENDERER_SAFE_SUBFOLDER,
            **model_common,
        ),
    }
    if adapter.input_kind == "text":
        components.update(
            {
                "text_encoder": CLIPTextModelWithProjection.from_pretrained(
                    repository,
                    subfolder="text_encoder",
                    **model_common,
                ),
                "tokenizer": CLIPTokenizer.from_pretrained(repository, subfolder="tokenizer", **common),
            }
        )
    else:
        components.update(
            {
                "image_encoder": CLIPVisionModel.from_pretrained(
                    repository,
                    subfolder="image_encoder",
                    **model_common,
                ),
                "image_processor": CLIPImageProcessor.from_pretrained(
                    repository,
                    subfolder="image_processor",
                    **common,
                ),
            }
        )
    return components


class LoadPipeline(NodeBase):
    """Load a reviewed generic Diffusers 3D pipeline."""

    label = "Load Diffusers 3D Pipeline"
    category = "Diffusers 3D"
    resizable = True
    cache_ignored_params = frozenset({"mode", "three_d_contract"})
    params = {
        "pipeline": {
            "label": "Pipeline",
            "display": "output",
            "type": "three_d_diffusion_pipeline",
            "signal": {
                "direction": "output",
                "origin": "three_d_contract",
                "value": DEFAULT_THREE_D_CONTRACT,
            },
        },
        "model_id": {
            "label": "Model",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": SHAP_E_REPO},
            "fieldOptions": {
                "noValidation": True,
                "sources": ["hub"],
                "filter": {"hub": {"className": THREE_D_PIPELINE_CLASS_OPTIONS}},
            },
            "onChange": "update_three_d_contract",
        },
        "pipeline_class": {
            "label": "Pipeline Class",
            "type": "string",
            "options": THREE_D_PIPELINE_CLASS_OPTIONS,
            "default": "ShapEPipeline",
            "fieldOptions": {"noValidation": True},
            "onChange": "update_three_d_contract",
        },
        "mode": {
            "label": "Mode",
            "type": "string",
            "options": THREE_D_MODE_OPTIONS,
            "default": "text_to_3d",
            "fieldOptions": {"noValidation": True},
            "onChange": "update_three_d_contract",
        },
        "three_d_contract": {
            "label": "3D Contract",
            "type": "object",
            "default": DEFAULT_THREE_D_CONTRACT,
            "hidden": True,
        },
        "revision": {"label": "Revision", "type": "string", "default": SHAP_E_REVISION},
        "dtype": {
            "label": "DType",
            "type": "string",
            "options": ["float32", "float16", "bfloat16"],
            "default": "float16",
        },
        "device": {"label": "Device", "type": "string", "options": DEVICE_OPTIONS, "default": DEFAULT_DEVICE},
        "auto_offload": {"label": "Auto offload", "type": "bool", "default": True},
        "offload_mode": offload_mode_param(modes=DIRECT_THREE_D_OFFLOAD_MODES),
        "execution_recipe": {
            "label": "Execution Recipe",
            "display": "input",
            "type": "diffusers_execution_recipe",
        },
        "low_cpu_mem_usage": {"label": "Low CPU memory", "type": "bool", "default": True},
        "resolved_artifact": {"label": "Resolved Artifact", "display": "output", "type": "string"},
    }

    def update_three_d_contract(self, values, ref):
        values = values if isinstance(values, dict) else {}
        adapter = _require_adapter(values.get("pipeline_class"), values.get("mode"))
        selected_model = values.get("model_id")
        if selected_model in (None, "", {"source": "hub", "value": SHAP_E_REPO}):
            selected_model = {"source": "hub", "value": adapter.default_repo}
        selected_revision = values.get("revision")
        if not selected_revision or selected_revision == SHAP_E_REVISION:
            selected_revision = adapter.revision
        repository, revision = _require_exact_selection(
            selected_model,
            selected_revision,
            adapter=adapter,
        )
        contract = adapter.signal_value()
        model_selection = {"source": "hub", "value": repository}
        self.set_field_value(
            {
                "model_id": model_selection,
                "revision": revision,
                "three_d_contract": contract,
            }
        )
        self.set_field_params(
            "pipeline",
            {
                "signal": {
                    "direction": "output",
                    "origin": "three_d_contract",
                    "value": contract,
                }
            },
        )

    def __call__(self, **kwargs):
        adapter = _require_adapter(kwargs.get("pipeline_class"), kwargs.get("mode"))
        repository, revision = _require_exact_selection(
            kwargs.get("model_id"),
            kwargs.get("revision"),
            adapter=adapter,
        )
        values = dict(kwargs)
        values["model_id"] = {"source": "hub", "value": repository}
        values["revision"] = revision
        result = super().__call__(**values)
        pipeline = result.get("pipeline") if isinstance(result, dict) else None
        if pipeline is not None:
            _tag_pipeline(pipeline, adapter, repository, revision)
        return result

    def execute(self, **kwargs):
        adapter = _require_adapter(kwargs.get("pipeline_class"), kwargs.get("mode"))
        repository, revision = _require_exact_selection(
            kwargs.get("model_id"),
            kwargs.get("revision"),
            adapter=adapter,
        )

        import diffusers
        from modules.DiffusersRuntime.main import apply_execution_recipe_to_pipeline, loader_runtime_options

        dtype = str_to_dtype(kwargs.get("dtype") or "float16")
        recipe, device, offload_mode, recipe_load_kwargs = loader_runtime_options(
            kwargs,
            default_device=DEFAULT_DEVICE,
            default_offload_mode=OFFLOAD_MODE_SEQUENTIAL_CPU,
            direct_device_load=False,
        )
        if recipe_load_kwargs:
            raise ValueError(
                "The reviewed Shap-E safe-component assembly does not support on-load quantization or device maps."
            )
        self.progress(-1, phase="loading", message="Loading safe Shap-E components")
        with self.diffusers_loading_progress():
            components = _load_safe_shap_e_components(
                repository,
                revision,
                dtype=dtype,
                low_cpu_mem_usage=bool(kwargs.get("low_cpu_mem_usage", True)),
                adapter=adapter,
            )
            pipeline_class = getattr(diffusers, adapter.pipeline_class)
            pipeline = pipeline_class(**components)
        _tag_pipeline(pipeline, adapter, repository, revision)
        if recipe:
            apply_execution_recipe_to_pipeline(pipeline, recipe)
        self.progress(99, phase="component_placement", message=f"Applying {offload_mode} offload")
        apply_pipeline_offload(
            pipeline,
            mode=offload_mode,
            device=device,
            node_id=self.node_id,
            scope="diffusers-three-d",
            prefer_pipeline_group=False,
        )
        self.mm_add(pipeline, priority=2)
        return {"pipeline": pipeline, "resolved_artifact": repository}


class GenerateRenderedArtifact(NodeBase):
    """Generate a bounded rendered orbit from a generic 3D pipeline."""

    label = "Diffusers 3D Rendered Generate"
    category = "Diffusers 3D"
    resizable = True
    params = {
        "pipeline": {
            "label": "Pipeline",
            "display": "input",
            "type": "three_d_diffusion_pipeline",
            "required": True,
            "onSignal": [
                {"action": "value", "target": "three_d_contract"},
                {"action": "exec", "data": "update_three_d_contract"},
            ],
        },
        "three_d_contract": {
            "label": "3D Contract",
            "type": "object",
            "default": DEFAULT_THREE_D_CONTRACT,
            "hidden": True,
        },
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "reference_images": {
            "label": "Reference Image",
            "display": "input",
            "type": "image",
            "required": False,
            "hidden": True,
        },
        "seed": {"label": "Seed", "type": "int", "display": "random", "default": 0, "min": 0, "max": 4294967295},
        "num_inference_steps": {
            "label": "Steps",
            "display": "slider",
            "type": "int",
            "default": 64,
            "min": 1,
            "max": 128,
        },
        "guidance_scale": {
            "label": "Guidance",
            "display": "slider",
            "type": "float",
            "default": 15.0,
            "min": 0,
            "max": 20,
            "step": 0.1,
        },
        "frame_size": {
            "label": "Frame Size",
            "type": "int",
            "default": 256,
            "min": 64,
            "max": 256,
            "step": 8,
        },
        "video": {"label": "Rendered Orbit", "display": "output", "type": "video"},
        "width_out": {"label": "Width", "display": "output", "type": "int"},
        "height_out": {"label": "Height", "display": "output", "type": "int"},
        "frames_out": {"label": "Frames", "display": "output", "type": "int"},
    }

    def update_three_d_contract(self, values, ref):
        values = values if isinstance(values, dict) else {}
        contract = values.get("three_d_contract")
        if not isinstance(contract, dict):
            raise ValueError("The connected 3D pipeline published a stale or mismatched task contract.")
        pipeline_class = contract.get("pipelineClass")
        adapter = THREE_D_PIPELINE_ADAPTERS.get(pipeline_class)
        if adapter is None or contract != adapter.signal_value():
            raise ValueError("The connected 3D pipeline published a stale or mismatched task contract.")
        for field, params in contract["fieldParams"].items():
            self.set_field_params(field, params)

    def __call__(self, **kwargs):
        _validate_connected_pipeline(kwargs.get("pipeline"), kwargs.get("three_d_contract"))
        return super().__call__(**kwargs)

    def execute(self, **kwargs):
        pipeline = kwargs.get("pipeline")
        adapter = _validate_connected_pipeline(pipeline, kwargs.get("three_d_contract"))
        prompt = kwargs.get("prompt")
        reference_images = kwargs.get("reference_images")
        if adapter.input_kind == "text":
            if reference_images is not None and not (
                isinstance(reference_images, list) and not reference_images
            ):
                raise ValueError("Text-conditioned rendered 3D generation does not accept a reference image.")
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2048:
                raise ValueError("Rendered 3D generation requires a prompt from 1 through 2048 characters.")
            conditioned_input = prompt.strip()
        else:
            if isinstance(prompt, str) and prompt.strip():
                raise ValueError("Image-conditioned rendered 3D generation does not accept a text prompt.")
            images = reference_images if isinstance(reference_images, list) else [reference_images]
            if len(images) != 1 or images[0] is None:
                raise ValueError("Image-conditioned rendered 3D generation requires exactly one reference image.")
            image = images[0]
            size = getattr(image, "size", None)
            if (
                not isinstance(size, tuple)
                or len(size) != 2
                or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in size)
                or size[0] * size[1] > 4_194_304
            ):
                raise ValueError("The Shap-E reference image must contain at most 4,194,304 positive pixels.")
            conditioned_input = image
        seed = _bounded_int(kwargs.get("seed"), default=0, label="Seed", minimum=0, maximum=4294967295)
        steps = _bounded_int(
            kwargs.get("num_inference_steps"),
            default=adapter.default_steps,
            label="Inference steps",
            minimum=1,
            maximum=adapter.max_steps,
        )
        guidance = _bounded_float(
            kwargs.get("guidance_scale"),
            default=adapter.default_guidance,
            label="Guidance",
            minimum=0,
            maximum=20,
        )
        frame_size = _bounded_int(
            kwargs.get("frame_size"),
            default=adapter.default_frame_size,
            label="Frame size",
            minimum=64,
            maximum=adapter.max_frame_size,
        )
        if frame_size % 8:
            raise ValueError("Frame size must be divisible by 8.")

        import torch

        device = getattr(pipeline, "_execution_device", None) or getattr(pipeline, "device", None) or "cpu"
        try:
            generator = torch.Generator(device=device).manual_seed(seed)
        except Exception:
            generator = torch.Generator(device="cpu").manual_seed(seed)
        self.progress(-1, phase="generating", message="Generating Shap-E latent and rendered orbit")
        call_kwargs = {
            "num_images_per_prompt": 1,
            "num_inference_steps": steps,
            "generator": generator,
            "guidance_scale": guidance,
            "frame_size": frame_size,
            "output_type": "pil",
            "return_dict": True,
        }
        call_kwargs["prompt" if adapter.input_kind == "text" else "image"] = conditioned_input
        result = pipeline(**call_kwargs)
        images = getattr(result, "images", None)
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], list):
            raise ValueError("Shap-E did not return one rendered-orbit frame sequence.")
        frames = images[0]
        if not 1 <= len(frames) <= 120:
            raise ValueError("Shap-E returned an empty or unbounded rendered-orbit frame sequence.")
        for frame in frames:
            size = getattr(frame, "size", None)
            if size != (frame_size, frame_size):
                raise ValueError("Shap-E returned a rendered frame outside the reviewed square frame contract.")
        return {
            "video": frames,
            "width_out": frame_size,
            "height_out": frame_size,
            "frames_out": len(frames),
        }
