"""MoDiff-owned schema helpers for Modular Diffusers node metadata.

Adapted from Hugging Face Diffusers modular pipeline utilities under Apache-2.0.
The local copy gives MoDiff a stable, product-owned schema and Hub config format.
"""

import copy
import json
import logging
import os

# Simple typed wrapper for parameter overrides
from dataclasses import asdict, dataclass
from typing import Any

from huggingface_hub import create_repo, hf_hub_download, upload_file
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from diffusers.utils import HUGGINGFACE_CO_RESOLVE_ENDPOINT
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam, OutputParam


logger = logging.getLogger(__name__)


def _name_to_label(name: str) -> str:
    """Convert snake_case name to Title Case label."""
    return name.replace("_", " ").title()


# Template definitions for standard diffuser pipeline parameters
MODIFF_PARAM_TEMPLATES = {
    # Image I/O
    "image": {"label": "Image", "type": "image", "display": "input", "required_block_params": ["image"]},
    "images": {"label": "Images", "type": "image", "display": "output", "required_block_params": ["images"]},
    "control_image": {
        "label": "Control Image",
        "type": "image",
        "display": "input",
        "required_block_params": ["control_image"],
    },
    # Latents
    "latents": {"label": "Latents", "type": "latents", "display": "input", "required_block_params": ["latents"]},
    "image_latents": {
        "label": "Image Latents",
        "type": "latents",
        "display": "input",
        "required_block_params": ["image_latents"],
    },
    "first_frame_latents": {
        "label": "First Frame Latents",
        "type": "latents",
        "display": "input",
        "required_block_params": ["first_frame_latents"],
    },
    "latents_preview": {"label": "Latents Preview", "type": "latent", "display": "output"},
    # Image Latents with Strength
    "image_latents_with_strength": {
        "name": "image_latents",  # name is not same as template key
        "label": "Image Latents",
        "type": "latents",
        "display": "input",
        "onChange": {"false": ["height", "width"], "true": ["strength"]},
        "required_block_params": ["image_latents", "strength"],
    },
    # Embeddings
    "embeddings": {"label": "Text Embeddings", "type": "embeddings", "display": "output"},
    "image_embeds": {
        "label": "Image Embeddings",
        "type": "image_embeds",
        "display": "output",
        "required_block_params": ["image_embeds"],
    },
    # Text inputs
    "prompt": {
        "label": "Prompt",
        "type": "string",
        "display": "textarea",
        "default": "",
        "required_block_params": ["prompt"],
    },
    "negative_prompt": {
        "label": "Negative Prompt",
        "type": "string",
        "display": "textarea",
        "default": "",
        "required_block_params": ["negative_prompt"],
    },
    # Numeric params
    "guidance_scale": {
        "label": "Guidance Scale",
        "type": "float",
        "display": "slider",
        "default": 5.0,
        "min": 1.0,
        "max": 30.0,
        "step": 0.1,
    },
    "strength": {
        "label": "Strength",
        "type": "float",
        "default": 0.5,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "required_block_params": ["strength"],
    },
    "height": {
        "label": "Height",
        "type": "int",
        "default": 1024,
        "min": 64,
        "step": 8,
        "required_block_params": ["height"],
    },
    "width": {
        "label": "Width",
        "type": "int",
        "default": 1024,
        "min": 64,
        "step": 8,
        "required_block_params": ["width"],
    },
    "seed": {
        "label": "Seed",
        "type": "int",
        "default": 0,
        "min": 0,
        "max": 4294967295,
        "display": "random",
        "required_block_params": ["generator"],
    },
    "num_inference_steps": {
        "label": "Steps",
        "type": "int",
        "default": 25,
        "min": 1,
        "max": 100,
        "display": "slider",
        "required_block_params": ["num_inference_steps"],
    },
    "num_frames": {
        "label": "Frames",
        "type": "int",
        "default": 81,
        "min": 1,
        "max": 480,
        "display": "slider",
        "required_block_params": ["num_frames"],
    },
    "layers": {
        "label": "Layers",
        "type": "int",
        "default": 4,
        "min": 1,
        "max": 10,
        "display": "slider",
        "required_block_params": ["layers"],
    },
    "output_type": {
        "label": "Output Type",
        "type": "dropdown",
        "default": "np",
        "options": ["np", "pil", "pt"],
    },
    # ControlNet
    "controlnet_conditioning_scale": {
        "label": "Controlnet Conditioning Scale",
        "type": "float",
        "default": 0.5,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "required_block_params": ["controlnet_conditioning_scale"],
    },
    "control_guidance_start": {
        "label": "Control Guidance Start",
        "type": "float",
        "default": 0.0,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "required_block_params": ["control_guidance_start"],
    },
    "control_guidance_end": {
        "label": "Control Guidance End",
        "type": "float",
        "default": 1.0,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "required_block_params": ["control_guidance_end"],
    },
    # Video
    "videos": {"label": "Videos", "type": "video", "display": "output", "required_block_params": ["videos"]},
    # Models
    "vae": {"label": "VAE", "type": "diffusers_auto_model", "display": "input", "required_block_params": ["vae"]},
    "image_encoder": {
        "label": "Image Encoder",
        "type": "diffusers_auto_model",
        "display": "input",
        "required_block_params": ["image_encoder"],
    },
    "unet": {"label": "Denoise Model", "type": "diffusers_auto_model", "display": "input"},
    "scheduler": {"label": "Scheduler", "type": "diffusers_auto_model", "display": "input"},
    "controlnet": {
        "label": "ControlNet Model",
        "type": "diffusers_auto_model",
        "display": "input",
        "required_block_params": ["controlnet"],
    },
    "text_encoders": {
        "label": "Text Encoders",
        "type": "diffusers_auto_models",
        "display": "input",
        "required_block_params": ["text_encoder"],
    },
    # Bundles/Custom
    "controlnet_bundle": {
        "label": "ControlNet",
        "type": "custom_controlnet",
        "display": "input",
        "required_block_params": "controlnet_image",
    },
    "ip_adapter": {"label": "IP Adapter", "type": "custom_ip_adapter", "display": "input"},
    "guider": {
        "label": "Guider",
        "type": "custom_guider",
        "display": "input",
        "onChange": {False: ["guidance_scale"], True: []},
    },
    "doc": {"label": "Doc", "type": "string", "display": "output"},
}


class MoDiffParamMeta(type):
    """Metaclass that enables MoDiffParam.template_name(**overrides) syntax."""

    def __getattr__(cls, name: str):
        if name in MODIFF_PARAM_TEMPLATES:

            def factory(default=None, **overrides):
                template = MODIFF_PARAM_TEMPLATES[name]
                # Use template's name if specified, otherwise use the key
                params = {"name": template.get("name", name), **template, **overrides}
                if default is not None:
                    params["default"] = default
                return cls(**params)

            return factory

        raise AttributeError(f"type object 'MoDiffParam' has no attribute '{name}'")


@dataclass(frozen=True)
class MoDiffParam(metaclass=MoDiffParamMeta):
    """
        Parameter definition for MoDiff nodes.

        Usage:
    ```python
        # From template (standard diffuser params)
        MoDiffParam.seed()
        MoDiffParam.prompt(default="a cat")
        MoDiffParam.latents(display="output")

        # Generic inputs (for custom blocks)
        MoDiffParam.Input.slider("my_scale", default=1.0, min=0.0, max=2.0)
        MoDiffParam.Input.dropdown("mode", options=["fast", "slow"])

        # Generic outputs
        MoDiffParam.Output.image("result_images")

        # Fully custom
        MoDiffParam(name="custom", label="Custom", type="float", default=0.5)
    ```
    """

    name: str
    label: str
    type: str
    display: str | None = None
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: Any = None
    value: Any = None
    fieldOptions: dict[str, Any] | None = None
    onChange: Any = None
    onSignal: Any = None
    required_block_params: str | list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for MoDiff schema, excluding None values and internal fields."""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None and k not in ("name", "required_block_params")}

    # =========================================================================
    # Input: Generic input parameter factories (for custom blocks)
    # =========================================================================
    class Input:
        """input UI elements for custom blocks."""

        @classmethod
        def image(cls, name: str) -> "MoDiffParam":
            """image input."""
            return MoDiffParam(name=name, label=_name_to_label(name), type="image", display="input")

        @classmethod
        def textbox(cls, name: str, default: str = "") -> "MoDiffParam":
            """text input as textarea."""
            return MoDiffParam(
                name=name, label=_name_to_label(name), type="string", display="textarea", default=default
            )

        @classmethod
        def dropdown(cls, name: str, options: list[str] = None, default: str = None) -> "MoDiffParam":
            """dropdown selection."""
            if options and not default:
                default = options[0]
            if not default:
                default = ""
            if not options:
                options = [default]
            return MoDiffParam(name=name, label=_name_to_label(name), type="string", options=options, value=default)

        @classmethod
        def slider(
            cls, name: str, default: float = 0, min: float = None, max: float = None, step: float = None
        ) -> "MoDiffParam":
            """slider input."""
            is_float = isinstance(default, float) or (step is not None and isinstance(step, float))
            param_type = "float" if is_float else "int"
            if min is None:
                min = default
            if max is None:
                max = default
            if step is None:
                step = 0.01 if is_float else 1
            return MoDiffParam(
                name=name,
                label=_name_to_label(name),
                type=param_type,
                display="slider",
                default=default,
                min=min,
                max=max,
                step=step,
            )

        @classmethod
        def number(
            cls, name: str, default: float = 0, min: float = None, max: float = None, step: float = None
        ) -> "MoDiffParam":
            """number input (no slider)."""
            is_float = isinstance(default, float) or (step is not None and isinstance(step, float))
            param_type = "float" if is_float else "int"
            return MoDiffParam(
                name=name, label=_name_to_label(name), type=param_type, default=default, min=min, max=max, step=step
            )

        @classmethod
        def seed(cls, name: str = "seed", default: int = 0) -> "MoDiffParam":
            """seed input with randomize button."""
            return MoDiffParam(
                name=name,
                label=_name_to_label(name),
                type="int",
                display="random",
                default=default,
                min=0,
                max=4294967295,
            )

        @classmethod
        def checkbox(cls, name: str, default: bool = False) -> "MoDiffParam":
            """boolean checkbox."""
            return MoDiffParam(name=name, label=_name_to_label(name), type="boolean", value=default)

        @classmethod
        def custom_type(cls, name: str, type: str) -> "MoDiffParam":
            """custom type input for node connections."""
            return MoDiffParam(name=name, label=_name_to_label(name), type=type, display="input")

        @classmethod
        def model(cls, name: str) -> "MoDiffParam":
            """model input for diffusers components."""
            return MoDiffParam(name=name, label=_name_to_label(name), type="diffusers_auto_model", display="input")

    # =========================================================================
    # Output: Generic output parameter factories (for custom blocks)
    # =========================================================================
    class Output:
        """output UI elements for custom blocks."""

        @classmethod
        def image(cls, name: str) -> "MoDiffParam":
            """image output."""
            return MoDiffParam(name=name, label=_name_to_label(name), type="image", display="output")

        @classmethod
        def video(cls, name: str) -> "MoDiffParam":
            """video output."""
            return MoDiffParam(name=name, label=_name_to_label(name), type="video", display="output")

        @classmethod
        def text(cls, name: str) -> "MoDiffParam":
            """text output."""
            return MoDiffParam(name=name, label=_name_to_label(name), type="string", display="output")

        @classmethod
        def custom_type(cls, name: str, type: str) -> "MoDiffParam":
            """custom type output for node connections."""
            return MoDiffParam(name=name, label=_name_to_label(name), type=type, display="output")

        @classmethod
        def model(cls, name: str) -> "MoDiffParam":
            """model output for diffusers components."""
            return MoDiffParam(name=name, label=_name_to_label(name), type="diffusers_auto_model", display="output")


def input_param_to_modiff_param(input_param: "InputParam") -> MoDiffParam:
    """
    Convert an InputParam to a MoDiffParam using metadata.

    Args:
        input_param: An InputParam with optional metadata containing either:
            - {"modiff": "<type>"} for simple types (image, textbox, slider, etc.)
            - {"modiff": MoDiffParam(...)} for full control over UI configuration

    Returns:
        MoDiffParam instance
    """
    name = input_param.name
    metadata = input_param.metadata
    modiff_value = metadata.get("modiff") if metadata else None
    default = input_param.default

    # If it's already a MoDiffParam, return it directly
    if isinstance(modiff_value, MoDiffParam):
        return modiff_value

    modiff_type = modiff_value

    if modiff_type == "image":
        return MoDiffParam.Input.image(name)
    elif modiff_type == "textbox":
        return MoDiffParam.Input.textbox(name, default=default or "")
    elif modiff_type == "dropdown":
        return MoDiffParam.Input.dropdown(name, default=default or "")
    elif modiff_type == "slider":
        return MoDiffParam.Input.slider(name, default=default or 0)
    elif modiff_type == "number":
        return MoDiffParam.Input.number(name, default=default or 0)
    elif modiff_type == "seed":
        return MoDiffParam.Input.seed(name, default=default or 0)
    elif modiff_type == "checkbox":
        return MoDiffParam.Input.checkbox(name, default=default or False)
    elif modiff_type == "model":
        return MoDiffParam.Input.model(name)
    else:
        # None or unknown -> custom
        return MoDiffParam.Input.custom_type(name, type="custom")


def output_param_to_modiff_param(output_param: "OutputParam") -> MoDiffParam:
    """
    Convert an OutputParam to a MoDiffParam using metadata.

    Args:
        output_param: An OutputParam with optional metadata={"modiff": "<type>"} where type is one of:
            image, video, text, model. If metadata is None or unknown, maps to "custom".

    Returns:
        MoDiffParam instance
    """
    name = output_param.name
    metadata = output_param.metadata
    modiff_type = metadata.get("modiff") if metadata else None

    if modiff_type == "image":
        return MoDiffParam.Output.image(name)
    elif modiff_type == "video":
        return MoDiffParam.Output.video(name)
    elif modiff_type == "text":
        return MoDiffParam.Output.text(name)
    elif modiff_type == "model":
        return MoDiffParam.Output.model(name)
    else:
        # None or unknown -> custom
        return MoDiffParam.Output.custom_type(name, type="custom")


DEFAULT_NODE_SPECS = {
    "controlnet": None,
    "denoise": {
        "inputs": [
            MoDiffParam.embeddings(display="input"),
            MoDiffParam.width(),
            MoDiffParam.height(),
            MoDiffParam.seed(),
            MoDiffParam.num_inference_steps(),
            MoDiffParam.num_frames(),
            MoDiffParam.guidance_scale(),
            MoDiffParam.strength(),
            MoDiffParam.image_latents_with_strength(),
            MoDiffParam.image_latents(),
            MoDiffParam.first_frame_latents(),
            MoDiffParam.controlnet_bundle(display="input"),
        ],
        "model_inputs": [
            MoDiffParam.unet(),
            MoDiffParam.guider(),
            MoDiffParam.scheduler(),
        ],
        "outputs": [
            MoDiffParam.latents(display="output"),
            MoDiffParam.latents_preview(),
            MoDiffParam.doc(),
        ],
        "required_inputs": ["embeddings"],
        "required_model_inputs": ["unet", "scheduler"],
        "block_name": "denoise",
    },
    "vae_encoder": {
        "inputs": [
            MoDiffParam.image(),
        ],
        "model_inputs": [
            MoDiffParam.vae(),
        ],
        "outputs": [
            MoDiffParam.image_latents(display="output"),
            MoDiffParam.doc(),
        ],
        "required_inputs": ["image"],
        "required_model_inputs": ["vae"],
        "block_name": "vae_encoder",
    },
    "text_encoder": {
        "inputs": [
            MoDiffParam.prompt(),
            MoDiffParam.negative_prompt(),
        ],
        "model_inputs": [
            MoDiffParam.text_encoders(),
        ],
        "outputs": [
            MoDiffParam.embeddings(display="output"),
            MoDiffParam.doc(),
        ],
        "required_inputs": ["prompt"],
        "required_model_inputs": ["text_encoders"],
        "block_name": "text_encoder",
    },
    "decoder": {
        "inputs": [
            MoDiffParam.latents(display="input"),
        ],
        "model_inputs": [
            MoDiffParam.vae(),
        ],
        "outputs": [
            MoDiffParam.images(),
            MoDiffParam.videos(),
            MoDiffParam.doc(),
        ],
        "required_inputs": ["latents"],
        "required_model_inputs": ["vae"],
        "block_name": "decode",
    },
}


def mark_required(label: str, marker: str = " *") -> str:
    """Add required marker to label if not already present."""
    if label.endswith(marker):
        return label
    return f"{label}{marker}"


def node_spec_to_modiff_dict(node_spec: dict[str, Any], node_type: str) -> dict[str, Any]:
    """
    Convert a node spec dict into MoDiff format.

    A node spec is how we define a MoDiff diffusers node in code. This function converts it into the `params` map
    format that MoDiff UI expects.

    The `params` map is a dict where keys are parameter names and values are UI configuration:
        ```python
        {"seed": {"label": "Seed", "type": "int", "default": 0}}
        ```

    For Modular MoDiff nodes, we need to distinguish:
        - `inputs`: Pipeline inputs (e.g., seed, prompt, image)
        - `mode…266 tokens truncated…  - `block_name`: The backend block name
            - `node_type`: The node type

    Example:
        ```python
        node_spec = {
            "inputs": [MoDiffParam.seed(), MoDiffParam.prompt()],
            "model_inputs": [MoDiffParam.unet()],
            "outputs": [MoDiffParam.latents(display="output")],
            "required_inputs": ["prompt"],
            "required_model_inputs": ["unet"],
            "block_name": "denoise",
        }

        result = node_spec_to_modiff_dict(node_spec, "denoise")
        # Returns:
        # {
        #     "params": {
        #         "seed": {"label": "Seed", "type": "int", "default": 0},
        #         "prompt": {"label": "Prompt *", "type": "string", "default": ""},  # * marks required
        #         "unet": {"label": "Denoise Model *", "type": "diffusers_auto_model", "display": "input"},
        #         "latents": {"label": "Latents", "type": "latents", "display": "output"},
        #     },
        #     "input_names": ["seed", "prompt"],
        #     "model_input_names": ["unet"],
        #     "output_names": ["latents"],
        #     "block_name": "denoise",
        #     "node_type": "denoise",
        # }
        ```
    """
    params = {}
    input_names = []
    model_input_names = []
    output_names = []

    required_inputs = node_spec.get("required_inputs", [])
    required_model_inputs = node_spec.get("required_model_inputs", [])

    # Process inputs
    for p in node_spec.get("inputs", []):
        param_dict = p.to_dict()
        if p.name in required_inputs:
            param_dict["label"] = mark_required(param_dict["label"])
        params[p.name] = param_dict
        input_names.append(p.name)

    # Process model_inputs
    for p in node_spec.get("model_inputs", []):
        param_dict = p.to_dict()
        if p.name in required_model_inputs:
            param_dict["label"] = mark_required(param_dict["label"])
        params[p.name] = param_dict
        model_input_names.append(p.name)

    # Process outputs: add a prefix to the output name if it already exists as an input
    for p in node_spec.get("outputs", []):
        if p.name in input_names:
            # rename to out_<name>
            output_name = f"out_{p.name}"
        else:
            output_name = p.name
        params[output_name] = p.to_dict()
        output_names.append(output_name)

    return {
        "params": params,
        "input_names": input_names,
        "model_input_names": model_input_names,
        "output_names": output_names,
        "block_name": node_spec.get("block_name"),
        "node_type": node_type,
    }


class MoDiffPipelineConfig:
    """
    Configuration for an entire MoDiff pipeline containing multiple nodes.

    Accepts node specs as dicts with inputs/model_inputs/outputs lists of MoDiffParam, converts them to MoDiff-ready
    format, and handles save/load to Hub.

    Example:
        ```python
        config = MoDiffPipelineConfig(
            node_specs={
                "denoise": {
                    "inputs": [MoDiffParam.seed(), MoDiffParam.prompt()],
                    "model_inputs": [MoDiffParam.unet()],
                    "outputs": [MoDiffParam.latents(display="output")],
                    "required_inputs": ["prompt"],
                    "required_model_inputs": ["unet"],
                    "block_name": "denoise",
                },
                "decoder": {
                    "inputs": [MoDiffParam.latents(display="input")],
                    "outputs": [MoDiffParam.images()],
                    "block_name": "decoder",
                },
            },
            label="My Pipeline",
            default_repo="user/my-pipeline",
            default_dtype="float16",
        )

        # Access MoDiff format dict
        denoise = config.node_params["denoise"]
        input_names = denoise["input_names"]
        params = denoise["params"]

        # Save to Hub
        config.save("./my_config", push_to_hub=True, repo_id="user/my-pipeline")

        # Load from Hub
        loaded = MoDiffPipelineConfig.load("user/my-pipeline")
        ```
    """

    config_name = "modiff_pipeline_config.json"

    def __init__(
        self,
        node_specs: dict[str, dict[str, Any] | None],
        label: str = "",
        default_repo: str = "",
        default_dtype: str = "",
    ):
        """
        Args:
            node_specs: Dict mapping node_type to node spec or None.
                        Node spec has: inputs, model_inputs, outputs, required_inputs, required_model_inputs,
                        block_name (all optional)
            label: Human-readable label for the pipeline
            default_repo: Default HuggingFace repo for this pipeline
            default_dtype: Default dtype (e.g., "float16", "bfloat16")
        """
        # Convert all node specs to MoDiff format immediately
        self.node_specs = node_specs

        self.label = label
        self.default_repo = default_repo
        self.default_dtype = default_dtype

    @property
    def node_params(self) -> dict[str, Any]:
        """Lazily compute node_params from node_specs."""
        if self.node_specs is None:
            return self._node_params

        params = {}
        for node_type, spec in self.node_specs.items():
            if spec is None:
                params[node_type] = None
            else:
                params[node_type] = node_spec_to_modiff_dict(spec, node_type)
        return params

    def __repr__(self) -> str:
        lines = [
            f"MoDiffPipelineConfig(label={self.label!r}, default_repo={self.default_repo!r}, default_dtype={self.default_dtype!r})"
        ]
        for node_type, spec in self.node_specs.items():
            if spec is None:
                lines.append(f"  {node_type}: None")
            else:
                inputs = [p.name for p in spec.get("inputs", [])]
                model_inputs = [p.name for p in spec.get("model_inputs", [])]
                outputs = [p.name for p in spec.get("outputs", [])]
                lines.append(f"  {node_type}:")
                lines.append(f"    inputs: {inputs}")
                lines.append(f"    model_inputs: {model_inputs}")
                lines.append(f"    outputs: {outputs}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "label": self.label,
            "default_repo": self.default_repo,
            "default_dtype": self.default_dtype,
            "node_params": self.node_params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MoDiffPipelineConfig":
        """
        Create from a dictionary (loaded from JSON).

        Note: The modiff_params are already in MoDiff format when loading from JSON.
        """
        instance = cls.__new__(cls)
        instance.node_specs = None
        instance._node_params = data.get("node_params", {})
        instance.label = data.get("label", "")
        instance.default_repo = data.get("default_repo", "")
        instance.default_dtype = data.get("default_dtype", "")
        return instance

    def to_json_string(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n"

    def to_json_file(self, json_file_path: str | os.PathLike):
        """Save to a JSON file."""
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    @classmethod
    def from_json_file(cls, json_file_path: str | os.PathLike) -> "MoDiffPipelineConfig":
        """Load from a JSON file."""
        with open(json_file_path, "r", encoding="utf-8") as reader:
            data = json.load(reader)
        return cls.from_dict(data)

    def save(self, save_directory: str | os.PathLike, push_to_hub: bool = False, **kwargs):
        """Save the modiff pipeline config to a directory."""
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)
        output_path = os.path.join(save_directory, self.config_name)
        self.to_json_file(output_path)
        logger.info(f"Pipeline config saved to {output_path}")

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            private = kwargs.pop("private", None)
            create_pr = kwargs.pop("create_pr", False)
            token = kwargs.pop("token", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id

            upload_file(
                path_or_fileobj=output_path,
                path_in_repo=self.config_name,
                repo_id=repo_id,
                token=token,
                commit_message=commit_message or "Upload MoDiffPipelineConfig",
                create_pr=create_pr,
            )
            logger.info(f"Pipeline config pushed to hub: {repo_id}")

    @classmethod
    def load(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        **kwargs,
    ) -> "MoDiffPipelineConfig":
        """Load a pipeline config from a local path or Hugging Face Hub."""
        cache_dir = kwargs.pop("cache_dir", None)
        local_dir = kwargs.pop("local_dir", None)
        local_dir_use_symlinks = kwargs.pop("local_dir_use_symlinks", "auto")
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        if os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, cls.config_name)
            if not os.path.isfile(config_file):
                raise EnvironmentError(f"No file named {cls.config_name} found in {pretrained_model_name_or_path}")
        else:
            try:
                config_file = hf_hub_download(
                    pretrained_model_name_or_path,
                    filename=cls.config_name,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    local_dir=local_dir,
                    local_dir_use_symlinks=local_dir_use_symlinks,
                )
            except RepositoryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier"
                    " listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a"
                    " token having permission to this repo with `token` or log in with `hf auth login`."
                )
            except RevisionNotFoundError:
                raise EnvironmentError(
                    f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for"
                    " this model name. Check the model page at"
                    f" 'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
                )
            except EntryNotFoundError:
                raise EnvironmentError(
                    f"{pretrained_model_name_or_path} does not appear to have a file named {cls.config_name}."
                )
            except HfHubHTTPError as err:
                raise EnvironmentError(
                    "There was a specific connection error when trying to load"
                    f" {pretrained_model_name_or_path}:\n{err}"
                )
            except ValueError:
                raise EnvironmentError(
                    f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it"
                    f" in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a"
                    f" directory containing a {cls.config_name} file.\nCheckout your internet connection or see how to"
                    " run the library in offline mode at"
                    " 'https://huggingface.co/docs/diffusers/installation#offline-mode'."
                )
            except EnvironmentError:
                raise EnvironmentError(
                    f"Can't load config for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                    "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                    f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a {cls.config_name} file"
                )

        try:
            return cls.from_json_file(config_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(f"The config file at '{config_file}' is not a valid JSON file.")

    @classmethod
    def from_blocks(
        cls,
        blocks,
        template: dict[str, dict[str, Any]] | None = None,
        label: str = "",
        default_repo: str = "",
        default_dtype: str = "bfloat16",
    ) -> "MoDiffPipelineConfig":
        """
        Create MoDiffPipelineConfig by matching template against actual pipeline blocks.
        """
        if template is None:
            template = DEFAULT_NODE_SPECS

        sub_block_map = dict(blocks.sub_blocks)

        def filter_spec_for_block(template_spec: dict[str, Any], block) -> dict[str, Any] | None:
            """Filter template spec params based on what the block actually supports."""
            block_input_names = set(block.input_names)
            block_output_names = set(block.intermediate_output_names)
            block_component_names = set(block.component_names)

            filtered_inputs = [
                p
                for p in template_spec.get("inputs", [])
                if p.required_block_params is None
                or all(name in block_input_names for name in p.required_block_params)
            ]
            filtered_model_inputs = [
                p
                for p in template_spec.get("model_inputs", [])
                if p.required_block_params is None
                or all(name in block_component_names for name in p.required_block_params)
            ]
            filtered_outputs = [
                p
                for p in template_spec.get("outputs", [])
                if p.required_block_params is None
                or all(name in block_output_names for name in p.required_block_params)
            ]

            filtered_input_names = {p.name for p in filtered_inputs}
            filtered_model_input_names = {p.name for p in filtered_model_inputs}

            filtered_required_inputs = [
                r for r in template_spec.get("required_inputs", []) if r in filtered_input_names
            ]
            filtered_required_model_inputs = [
                r for r in template_spec.get("required_model_inputs", []) if r in filtered_model_input_names
            ]

            return {
                "inputs": filtered_inputs,
                "model_inputs": filtered_model_inputs,
                "outputs": filtered_outputs,
                "required_inputs": filtered_required_inputs,
                "required_model_inputs": filtered_required_model_inputs,
                "block_name": template_spec.get("block_name"),
            }

        # Build node specs
        node_specs = {}
        for node_type, template_spec in template.items():
            if template_spec is None:
                node_specs[node_type] = None
                continue

            block_name = template_spec.get("block_name")
            if block_name is None or block_name not in sub_block_map:
                node_specs[node_type] = None
                continue

            node_specs[node_type] = filter_spec_for_block(template_spec, sub_block_map[block_name])

        return cls(
            node_specs=node_specs,
            label=label or getattr(blocks, "model_name", ""),
            default_repo=default_repo,
            default_dtype=default_dtype,
        )

    @classmethod
    def from_custom_block(
        cls,
        block,
        node_label: str = None,
        input_types: dict[str, Any] | None = None,
        output_types: dict[str, Any] | None = None,
    ) -> "MoDiffPipelineConfig":
        """
        Create a MoDiffPipelineConfig from a custom block.

        Args:
            block: A block instance with `inputs`, `outputs`, and `expected_components`/`component_names` properties.
                Each InputParam/OutputParam should have metadata={"modiff": "<type>"} where type is one of: image,
                video, text, checkbox, number, slider, dropdown, model. If metadata is None, maps to "custom".
            node_label: The display label for the node. Defaults to block class name with spaces.
            input_types:
                Optional dict mapping input param names to modiff types. Overrides the block's metadata if provided.
                Example: {"prompt": "textbox", "image": "image"}
            output_types:
                Optional dict mapping output param names to modiff types. Overrides the block's metadata if provided.
                Example: {"prompt": "text", "images": "image"}

        Returns:
            MoDiffPipelineConfig instance
        """
        if node_label is None:
            class_name = block.__class__.__name__
            node_label = "".join([" " + c if c.isupper() else c for c in class_name]).strip()

        if input_types is None:
            input_types = {}
        if output_types is None:
            output_types = {}

        inputs = []
        model_inputs = []
        outputs = []

        # Process block inputs
        for input_param in block.inputs:
            if input_param.name is None:
                continue
            if input_param.name in input_types:
                input_param = copy.copy(input_param)
                input_param.metadata = {"modiff": input_types[input_param.name]}
            print(f" processing input: {input_param.name}, metadata: {input_param.metadata}")
            inputs.append(input_param_to_modiff_param(input_param))

        # Process block outputs
        for output_param in block.outputs:
            if output_param.name is None:
                continue
            if output_param.name in output_types:
                output_param = copy.copy(output_param)
                output_param.metadata = {"modiff": output_types[output_param.name]}
            outputs.append(output_param_to_modiff_param(output_param))

        # Process expected components (all map to model inputs)
        component_names = block.component_names
        for component_name in component_names:
            model_inputs.append(MoDiffParam.Input.model(component_name))

        # Always add doc output
        outputs.append(MoDiffParam.doc())

        node_spec = {
            "inputs": inputs,
            "model_inputs": model_inputs,
            "outputs": outputs,
            "required_inputs": [],
            "required_model_inputs": [],
            "block_name": "custom",
        }

        return cls(
            node_specs={"custom": node_spec},
            label=node_label,
        )
