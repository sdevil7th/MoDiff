# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import importlib
import json
import logging
import time

import torch
from PIL import Image

from modiff.NodeBase import NodeBase

from . import MESSAGE_DURATION, components
from .modular_utils import (
    DummyCustomPipeline,
    pipeline_class_from_runtime_inputs,
    pipeline_class_to_modiff_node_config,
)
from .utils import collect_model_ids


logger = logging.getLogger("modiff")


def sanitized_tensor_summary(value):
    """Return JSON-safe tensor metadata without retaining or serializing data."""
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        for item in value.values():
            summary = sanitized_tensor_summary(item)
            if summary:
                return summary
    if isinstance(value, (list, tuple)):
        for item in value:
            summary = sanitized_tensor_summary(item)
            if summary:
                return summary
    return None


def flatten_pil_images(value):
    """Flatten nested Diffusers image batches without changing non-image outputs."""
    if isinstance(value, Image.Image):
        return [value]
    if isinstance(value, (list, tuple)):
        images = []
        for item in value:
            flattened = flatten_pil_images(item)
            if flattened is None:
                return None
            images.extend(flattened)
        return images
    return None


def prepare_image_for_vae_pipeline(image, pipeline_class):
    """Apply pipeline-specific source-channel contracts before VAE encoding.

    Qwen Image Layered's VAE is trained for RGBA input (`in_channels=4`) and
    the upstream Diffusers example explicitly converts source media to RGBA.
    MoDiff's shared image loader normally returns RGB PIL images, so preserving
    that generic value here would fail only after the expensive model load.
    Keep the adaptation at the VAE boundary and leave every other pipeline
    unchanged.
    """

    pipeline_name = getattr(pipeline_class, "__name__", "")
    if pipeline_name not in {"QwenImageLayeredModularPipeline", "QwenImageLayeredPipeline"}:
        return image
    if isinstance(image, Image.Image):
        return image if image.mode == "RGBA" else image.convert("RGBA")
    if isinstance(image, list):
        return [prepare_image_for_vae_pipeline(item, pipeline_class) for item in image]
    if isinstance(image, tuple):
        return tuple(prepare_image_for_vae_pipeline(item, pipeline_class) for item in image)
    return image


# YiYi Notes: this is not working for qwen/flux as latents needs to be unpacked first
class LatentsPreview(NodeBase):
    label = "Latents Preview"
    category = "image"
    resizable = True
    params = {
        "latents": {"label": "Latents", "display": "input", "type": "latent"},
        "image": {"label": "Image", "display": "output", "type": "image", "hidden": True},
        "preview": {"display": "ui_image", "dataSource": "image"},
    }

    def execute(self, latents):
        latent_rgb_factors = [
            [0.3920, 0.4054, 0.4549],
            [-0.2634, -0.0196, 0.0653],
            [0.0568, 0.1687, -0.0755],
            [-0.3112, -0.2359, -0.2076],
        ]

        image = None

        if latents is not None:
            latent_rgb_factors = torch.tensor(latent_rgb_factors, dtype=latents.dtype).to(device=latents.device)
            latent_image = latents.squeeze(0).permute(1, 2, 0) @ latent_rgb_factors
            latents_ubyte = ((latent_image + 1.0) / 2.0).clamp(0, 1).mul(0xFF)
            denoised_image = latents_ubyte.byte().cpu().numpy()
            image = Image.fromarray(denoised_image)
            image = image.resize((image.width * 2, image.height * 2), resample=Image.Resampling.BICUBIC)

        return {"image": image}


class DecodeLatents(NodeBase):
    label = "Decode Latents"
    category = "sampler"
    resizable = True
    skipParamsCheck = True
    node_type = "decoder"
    params = {
        "vae": {
            "label": "VAE *",
            "display": "input",
            "type": "diffusers_auto_model",
            "onSignal": "update_node",
        },
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._model_type = ""
        self._pipeline_class = None

    def update_node(self, values, ref):
        node_params = {}
        model_type = self.get_signal_value("vae")

        if self._model_type == model_type:
            return None

        if model_type is None or model_type == "" or model_type == "DummyCustomPipeline":
            self._pipeline_class = DummyCustomPipeline
        else:
            diffusers_module = importlib.import_module("diffusers")
            self._pipeline_class = getattr(diffusers_module, model_type)

        self._model_type = model_type

        _, node_config = pipeline_class_to_modiff_node_config(self._pipeline_class, self.node_type)

        if node_config is None:
            self.send_node_definition(node_params)
            return

        node_params_to_update = node_config["params"]
        node_params_to_update.pop("vae", None)
        node_params.update(**node_params_to_update)
        self.send_node_definition(node_params)

    def execute(self, **kwargs):
        kwargs = dict(kwargs)
        self._pipeline_class = pipeline_class_from_runtime_inputs(self._pipeline_class, kwargs)

        # 1. Get node config
        blocks, node_config = pipeline_class_to_modiff_node_config(self._pipeline_class, self.node_type)

        # 2. Create pipeline
        repo_id = None
        if (vae := kwargs.get("vae")) and "repo_id" in vae:
            repo_id = vae["repo_id"]

        if repo_id is None:
            self.notify(
                "You have to connect the vae",
                variant="error",
                persist=False,
                autoHideDuration=MESSAGE_DURATION,
            )
            return None

        self._pipeline = blocks.init_pipeline(repo_id, components_manager=components)

        # 3. Cast parameters to the types expected by the modular pipeline.
        for param_name, param_config in node_config["params"].items():
            if param_name in kwargs and kwargs[param_name] is not None:
                param_type = param_config.get("type", None)
                if param_type == "float":
                    kwargs[param_name] = float(kwargs[param_name])
                elif param_type == "int":
                    kwargs[param_name] = int(kwargs[param_name])

        # 4. Update components
        expected_component_names = blocks.component_names
        model_input_names = node_config["model_input_names"]
        model_ids = collect_model_ids(
            kwargs,
            target_key_names=model_input_names,
            target_model_names=expected_component_names,
        )

        if model_ids:
            components_to_update = components.get_components_by_ids(ids=model_ids, return_dict_with_names=True)
            if components_to_update:
                self._pipeline.update_components(**components_to_update)

        # 5. Compile runtime inputs from kwargs based on node_config["input_names"]
        node_kwargs = {}
        input_names = node_config["input_names"]

        for name in input_names:
            if name not in kwargs:
                continue
            value = kwargs.get(name)

            if isinstance(value, dict) and name not in blocks.input_names:
                for k, v in value.items():
                    if k in blocks.input_names:
                        node_kwargs[k] = v
            elif name in blocks.input_names:
                node_kwargs[name] = value

        # 6. Run the pipeline
        try:
            node_output_state = self._pipeline(**node_kwargs)
        except ValueError as e:
            self.notify(str(e), variant="error", persist=False, autoHideDuration=MESSAGE_DURATION)
            raise

        # 7. Prepare outputs based on node_config["output_names"]
        outputs = {}
        output_names = node_config["output_names"].copy()

        for name in output_names:
            if name == "doc":
                outputs["doc"] = self._pipeline.blocks.doc
            else:
                value = node_output_state.get(name)
                if name == "images":
                    flattened = flatten_pil_images(value)
                    if flattened is not None:
                        value = flattened[0] if len(flattened) == 1 else flattened
                outputs[name] = value

        return outputs


class ImageEncode(NodeBase):
    label = "Encode Image"
    category = "sampler"
    resizable = True
    skipParamsCheck = True
    node_type = "vae_encoder"
    params = {
        "vae": {"label": "VAE *", "display": "input", "type": "diffusers_auto_model", "onSignal": "update_node"},
        "encode_summary_data": {
            "label": "Encode summary",
            "display": "output",
            "type": "str",
            "hidden": True,
        },
        "encode_summary": {
            "label": "Encode summary",
            "display": "ui_text",
            "type": "text",
            "dataSource": "encode_summary_data",
            "hidden": True,
        },
    }

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._model_type = ""
        self._pipeline_class = None

    def update_node(self, values, ref):
        node_params = {}
        model_type = self.get_signal_value("vae")

        if self._model_type == model_type:
            return None

        if model_type is None or model_type == "" or model_type == "DummyCustomPipeline":
            self._pipeline_class = DummyCustomPipeline
        else:
            diffusers_module = importlib.import_module("diffusers")
            self._pipeline_class = getattr(diffusers_module, model_type)

        self._model_type = model_type

        _, node_config = pipeline_class_to_modiff_node_config(self._pipeline_class, self.node_type)

        if node_config is None:
            self.send_node_definition(node_params)
            return

        node_params_to_update = node_config["params"]
        node_params_to_update.pop("vae", None)
        node_params.update(**node_params_to_update)
        self.send_node_definition(node_params)

    def execute(self, **kwargs):
        encode_started_at = time.perf_counter()
        self.progress(
            0,
            phase="encoding",
            message="Encoding source image into latent representation",
            current_step=0,
            total_steps=1,
            elapsed_seconds=0.0,
        )
        kwargs = dict(kwargs)
        self._pipeline_class = pipeline_class_from_runtime_inputs(self._pipeline_class, kwargs)

        # 1. Get node config
        blocks, node_config = pipeline_class_to_modiff_node_config(self._pipeline_class, self.node_type)

        # 2. Create pipeline
        repo_id = None
        if (vae := kwargs.get("vae")) and "repo_id" in vae:
            repo_id = vae["repo_id"]

        if repo_id is None:
            self.notify(
                "You have to connect the vae",
                variant="error",
                persist=False,
                autoHideDuration=MESSAGE_DURATION,
            )
            return None

        self._pipeline = blocks.init_pipeline(repo_id, components_manager=components)

        # 3. Cast parameters to the types expected by the modular pipeline.
        for param_name, param_config in node_config["params"].items():
            if param_name in kwargs and kwargs[param_name] is not None:
                param_type = param_config.get("type", None)
                if param_type == "float":
                    kwargs[param_name] = float(kwargs[param_name])
                elif param_type == "int":
                    kwargs[param_name] = int(kwargs[param_name])

        # 4. Update components
        expected_component_names = blocks.component_names
        model_input_names = node_config["model_input_names"]
        model_ids = collect_model_ids(
            kwargs,
            target_key_names=model_input_names,
            target_model_names=expected_component_names,
        )

        if model_ids:
            components_to_update = components.get_components_by_ids(ids=model_ids, return_dict_with_names=True)
            if components_to_update:
                self._pipeline.update_components(**components_to_update)

        # 5. Compile runtime inputs from kwargs based on node_config["input_names"]
        node_kwargs = {}
        input_names = node_config["input_names"]

        for name in input_names:
            if name not in kwargs:
                continue
            value = kwargs.get(name)

            if isinstance(value, dict) and name not in blocks.input_names:
                for k, v in value.items():
                    if k in blocks.input_names:
                        node_kwargs[k] = v
            elif name in blocks.input_names:
                node_kwargs[name] = value

        if "image" in node_kwargs:
            node_kwargs["image"] = prepare_image_for_vae_pipeline(node_kwargs["image"], self._pipeline_class)

        # 6. Run the pipeline
        try:
            node_output_state = self._pipeline(**node_kwargs)
        except ValueError as e:
            self.notify(str(e), variant="error", persist=False, autoHideDuration=MESSAGE_DURATION)
            raise

        # 7. Prepare outputs based on node_config["output_names"]
        output_names = node_config["output_names"].copy()
        outputs = {}
        for name in output_names:
            if name == "doc":
                outputs["doc"] = self._pipeline.blocks.doc
            else:
                outputs[name] = node_output_state.get(name)

        tensor_summary = None
        for name in output_names:
            tensor_summary = sanitized_tensor_summary(outputs.get(name))
            if tensor_summary:
                break
        elapsed_seconds = round(max(0.0, time.perf_counter() - encode_started_at), 4)
        outputs["encode_summary_data"] = json.dumps({
            "schemaVersion": 1,
            "status": "encoded",
            "updatedAt": time.time(),
            "elapsedSeconds": elapsed_seconds,
            **(tensor_summary or {}),
        }, separators=(",", ":"))
        self.progress(
            100,
            phase="encoding",
            message="Encoded source image",
            current_step=1,
            total_steps=1,
            elapsed_seconds=elapsed_seconds,
        )

        return outputs
