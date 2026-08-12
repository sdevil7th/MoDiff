# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import json
import logging
import time

import torch
from PIL import Image
from diffusers.modular_pipelines import PipelineState

from modiff.NodeBase import NodeBase

from . import MESSAGE_DURATION, components
from .modular_utils import (
    modular_generator_from_seed,
    normalize_modular_runtime_params,
    normalize_modular_seed,
    pipeline_class_from_model_type,
    pipeline_class_from_runtime_inputs,
    reject_undeclared_modular_generator,
    require_modiff_node_contract,
)
from .route_state import (
    ROUTE_STATE_INPUT,
    ROUTE_STATE_OUTPUT,
    SUPPORTED_ROUTE_MODEL_TYPES,
    consume_decode_route_state,
    consume_wan_image_encoder_route_state,
    effective_modular_block_input,
    issue_encoder_route_state,
    issue_wan_vae_route_state,
    preflight_wan_vae_route_state,
    reject_route_reserved_inputs_before_identity_resolution,
    reject_route_reserved_inputs,
    resolve_managed_component_by_id,
    require_cataloged_wan_action_source,
    require_component_binding,
    require_route_state_shape_before_identity_resolution,
    route_contract_for_model_type,
    route_cache_params_equal,
    require_wan_video_processor,
    sdxl_vae_geometry_from_component,
    validate_sdxl_crop_overlay_inputs,
    validate_encoder_route_state,
    validate_wan_post_vae_route_state,
    validate_route_field_contract,
    wan_video_processor_config_seal,
)
from .utils import collect_model_ids


logger = logging.getLogger("modiff")


def require_exact_installed_component(pipeline, component_updates, expected_component_names, name):
    """Reject ambient ComponentsManager selection for an explicitly connected port."""

    if name not in expected_component_names:
        return None
    expected = component_updates.get(name)
    if expected is None:
        raise ValueError(f"The connected Modular {name} could not be resolved by its exact managed component ID.")
    if getattr(pipeline, name, None) is not expected:
        raise ValueError(f"The Modular pipeline did not install the exact connected {name} component.")
    return expected


def require_exact_resident_component(pipeline, component_input, name, *, label):
    """Resolve a cache-time component and reject ambient or replaced residents."""

    resolved = resolve_managed_component_by_id(components, component_input, label=label)
    if pipeline is None or getattr(pipeline, name, None) is not resolved:
        raise ValueError(f"The resident Modular pipeline does not hold the exact connected {name} component.")
    return resolved


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
        self._wan_decode_video_processor = None
        self._wan_decode_video_processor_config_seal = None

    def _cache_params_equal(self, previous, current):
        equal = route_cache_params_equal(previous, current, fallback=super()._cache_params_equal)
        if not equal or not isinstance(current, dict):
            return equal
        route_state = current.get(ROUTE_STATE_INPUT)
        if self._pipeline_class is None:
            return route_state is None
        model_type = getattr(self._pipeline_class, "__name__", "")
        if model_type not in SUPPORTED_ROUTE_MODEL_TYPES:
            return True
        if route_state is None:
            return False
        blocks, node_config = require_modiff_node_contract(self._pipeline_class, self.node_type)
        require_route_state_shape_before_identity_resolution(current)
        reject_route_reserved_inputs_before_identity_resolution(current)
        reject_route_reserved_inputs(current, model_type=model_type, action="decoder")
        binding = require_component_binding(
            current.get("vae"),
            label="VAE",
            expected_model_type=model_type,
            expected_role="vae",
        )
        resident_vae = None
        resident_geometry = (None, None)
        route_contract = route_contract_for_model_type(model_type)
        resident_video_processor = None
        if route_contract in {"sdxl", "wan_i2v"}:
            resident_vae = require_exact_resident_component(
                getattr(self, "_pipeline", None),
                current.get("vae"),
                "vae",
                label="Decode VAE",
            )
            if route_contract == "sdxl":
                resident_geometry = sdxl_vae_geometry_from_component(resident_vae)
            else:
                resident_video_processor = require_wan_video_processor(
                    getattr(getattr(self, "_pipeline", None), "video_processor", None)
                )
                if resident_video_processor is not self._wan_decode_video_processor:
                    return False
                if (
                    wan_video_processor_config_seal(resident_video_processor)
                    != self._wan_decode_video_processor_config_seal
                ):
                    raise ValueError(
                        "The Wan Decode video processor configuration changed after cached output publication."
                    )
        effective_latents = effective_modular_block_input(
            current,
            node_input_names=node_config["input_names"],
            block_input_names=blocks.input_names,
            target_name="latents",
        )
        consume_decode_route_state(
            route_state,
            binding=binding,
            model_type=model_type,
            latents=effective_latents,
            vae_component=resident_vae,
            vae_latent_channels=resident_geometry[0],
            vae_scale_factor=resident_geometry[1],
            video_processor=resident_video_processor,
            execution_device=(
                getattr(getattr(self, "_pipeline", None), "_execution_device", None)
                if route_contract == "wan_i2v"
                else None
            ),
            materialize_overlay=False,
        )
        return True

    def update_node(self, values, ref):
        node_params = {}
        model_type = self.get_signal_value("vae")

        if self._model_type == model_type:
            if not model_type or self._pipeline_class is None:
                return None
            _, node_config = require_modiff_node_contract(
                self._pipeline_class,
                self.node_type,
                resolve_blocks=False,
            )
            node_params_to_update = dict(node_config["params"])
            node_params_to_update.pop("vae", None)
            self.send_node_definition(node_params_to_update)
            return None

        if model_type is None or model_type == "":
            self._model_type = ""
            self._pipeline_class = None
            self.send_node_definition(node_params)
            return None
        try:
            self._pipeline_class = pipeline_class_from_model_type(model_type)
            _, node_config = require_modiff_node_contract(
                self._pipeline_class,
                self.node_type,
                resolve_blocks=False,
            )
        except ValueError:
            self._model_type = ""
            self._pipeline_class = None
            self.send_node_definition(node_params)
            raise
        self._model_type = model_type

        node_params_to_update = dict(node_config["params"])
        node_params_to_update.pop("vae", None)
        node_params.update(**node_params_to_update)
        self.send_node_definition(node_params)

    def execute(self, **kwargs):
        kwargs = dict(kwargs)
        require_route_state_shape_before_identity_resolution(kwargs)
        reject_route_reserved_inputs_before_identity_resolution(kwargs)
        identity_kwargs = {name: value for name, value in kwargs.items() if name != ROUTE_STATE_INPUT}
        self._pipeline_class = pipeline_class_from_runtime_inputs(self._pipeline_class, identity_kwargs)

        # 1. Get node config
        blocks, node_config = require_modiff_node_contract(self._pipeline_class, self.node_type)
        validate_route_field_contract(kwargs, node_config)

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

        route_state = kwargs.get(ROUTE_STATE_INPUT)
        if (
            getattr(self._pipeline_class, "__name__", "") in SUPPORTED_ROUTE_MODEL_TYPES
            and route_state is None
        ):
            raise ValueError(
                "Modular Decode requires the opaque route state emitted by its matching Denoise action."
            )
        route_values = None
        preinit_vae = None
        preinit_geometry = (None, None)
        decode_video_processor = None
        decode_video_processor_config_seal = None
        if route_state is not None:
            reject_route_reserved_inputs(
                kwargs,
                model_type=getattr(self._pipeline_class, "__name__", ""),
                action="decoder",
            )
            effective_latents = effective_modular_block_input(
                kwargs,
                node_input_names=node_config["input_names"],
                block_input_names=blocks.input_names,
                target_name="latents",
            )
            binding = require_component_binding(
                vae,
                label="VAE",
                expected_model_type=getattr(self._pipeline_class, "__name__", ""),
                expected_role="vae",
            )
            decode_contract = route_contract_for_model_type(getattr(self._pipeline_class, "__name__", ""))
            if decode_contract in {"sdxl", "wan_i2v"}:
                preinit_vae = resolve_managed_component_by_id(components, vae, label="Decode VAE")
                if decode_contract == "sdxl":
                    preinit_geometry = sdxl_vae_geometry_from_component(preinit_vae)
            route_values = consume_decode_route_state(
                route_state,
                binding=binding,
                model_type=getattr(self._pipeline_class, "__name__", ""),
                latents=effective_latents,
                vae_component=preinit_vae,
                vae_latent_channels=preinit_geometry[0],
                vae_scale_factor=preinit_geometry[1],
                materialize_overlay=False,
            )

        # Use the installed block contract and inject the already-managed VAE;
        # never re-read repository component hints in a downstream action.
        self._pipeline = blocks.init_pipeline(components_manager=components)

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

        components_to_update = {}
        if model_ids:
            components_to_update = components.get_components_by_ids(ids=model_ids, return_dict_with_names=True)
            if components_to_update:
                self._pipeline.update_components(**components_to_update)
        if route_state is not None:
            require_component_binding(
                vae,
                label="VAE",
                expected_model_type=getattr(self._pipeline_class, "__name__", ""),
                expected_token=binding,
                expected_role="vae",
            )
            installed_vae = require_exact_installed_component(
                self._pipeline,
                components_to_update,
                expected_component_names,
                "vae",
            )
            live_vae = installed_vae
            if installed_vae is not None:
                live_vae = require_exact_resident_component(
                    self._pipeline,
                    vae,
                    "vae",
                    label="Decode VAE",
                )
                if live_vae is not installed_vae:
                    raise ValueError("The connected Decode VAE changed during component installation.")
            if decode_contract == "sdxl":
                if installed_vae is not preinit_vae:
                    raise ValueError("The connected Decode VAE changed during pipeline initialization.")
                vae_latent_channels, vae_scale_factor = sdxl_vae_geometry_from_component(installed_vae)
                route_values = consume_decode_route_state(
                    route_state,
                    binding=binding,
                    model_type=getattr(self._pipeline_class, "__name__", ""),
                    latents=effective_latents,
                    vae_component=installed_vae,
                    vae_latent_channels=vae_latent_channels,
                    vae_scale_factor=vae_scale_factor,
                )
            elif decode_contract == "wan_i2v":
                if installed_vae is not preinit_vae:
                    raise ValueError("The connected Wan Decode VAE changed during pipeline initialization.")
                decode_video_processor = require_wan_video_processor(
                    getattr(self._pipeline, "video_processor", None)
                )
                decode_video_processor_config_seal = wan_video_processor_config_seal(decode_video_processor)
                route_values = consume_decode_route_state(
                    route_state,
                    binding=binding,
                    model_type=getattr(self._pipeline_class, "__name__", ""),
                    latents=effective_latents,
                    vae_component=installed_vae,
                    video_processor=decode_video_processor,
                    execution_device=self._pipeline._execution_device,
                    materialize_overlay=False,
                )

        # 5. Compile runtime inputs from kwargs based on node_config["input_names"]
        node_kwargs = {}
        input_names = node_config["input_names"]

        for name in input_names:
            if name == ROUTE_STATE_INPUT:
                continue
            if name not in kwargs:
                continue
            value = kwargs.get(name)

            if isinstance(value, dict) and name not in blocks.input_names:
                for k, v in value.items():
                    if k in blocks.input_names:
                        node_kwargs[k] = v
            elif name in blocks.input_names:
                node_kwargs[name] = value

        if route_values is not None and route_values["mask_overlay_kwargs"] is not None:
            if "mask_overlay_kwargs" not in blocks.input_names:
                raise ValueError("The selected Modular decoder does not expose the routed mask overlay input.")
            node_kwargs["mask_overlay_kwargs"] = route_values["mask_overlay_kwargs"]
        if route_values is not None and route_values["decode_inputs"] is not None:
            for name, value in route_values["decode_inputs"].items():
                if name not in blocks.input_names:
                    raise ValueError(f"The selected Modular decoder does not expose routed SDXL input '{name}'.")
                node_kwargs[name] = value

        # 6. Run the pipeline
        if route_state is not None and route_values["contract"] == "wan_i2v":
            if require_exact_resident_component(
                self._pipeline,
                vae,
                "vae",
                label="Decode VAE",
            ) is not installed_vae:
                raise ValueError("The connected Wan Decode VAE changed before upstream execution.")
            if getattr(self._pipeline, "video_processor", None) is not decode_video_processor:
                raise ValueError("The Wan Decode video processor changed before upstream execution.")
            if wan_video_processor_config_seal(decode_video_processor) != decode_video_processor_config_seal:
                raise ValueError("The Wan Decode video processor configuration changed before upstream execution.")
            consume_decode_route_state(
                route_state,
                binding=binding,
                model_type=getattr(self._pipeline_class, "__name__", ""),
                latents=effective_latents,
                vae_component=installed_vae,
                video_processor=decode_video_processor,
                execution_device=self._pipeline._execution_device,
                materialize_overlay=False,
            )
        try:
            if route_values is not None and route_values["contract"] == "qwen" and route_values["inpaint"]:
                node_output_state = self._pipeline(
                    state=PipelineState(values={"mask": True}),
                    **node_kwargs,
                )
            else:
                node_output_state = self._pipeline(**node_kwargs)
        except ValueError as e:
            self.notify(str(e), variant="error", persist=False, autoHideDuration=MESSAGE_DURATION)
            raise

        if route_state is not None:
            require_component_binding(
                vae,
                label="VAE",
                expected_model_type=getattr(self._pipeline_class, "__name__", ""),
                expected_token=binding,
                expected_role="vae",
            )
            installed_vae = require_exact_installed_component(
                self._pipeline,
                components_to_update,
                expected_component_names,
                "vae",
            )
            live_vae = installed_vae
            if installed_vae is not None:
                live_vae = require_exact_resident_component(
                    self._pipeline,
                    vae,
                    "vae",
                    label="Decode VAE",
                )
                if live_vae is not installed_vae:
                    raise ValueError("The connected Decode VAE changed during upstream execution.")
            post_geometry = (None, None)
            if route_values["contract"] == "sdxl":
                post_latent_channels, post_scale_factor = sdxl_vae_geometry_from_component(installed_vae)
                if (post_latent_channels, post_scale_factor) != (vae_latent_channels, vae_scale_factor):
                    raise ValueError("The connected Decode VAE geometry changed during upstream execution.")
                post_geometry = (post_latent_channels, post_scale_factor)
            elif route_values["contract"] == "wan_i2v":
                if getattr(self._pipeline, "video_processor", None) is not decode_video_processor:
                    raise ValueError("The Wan Decode video processor changed during upstream execution.")
                if wan_video_processor_config_seal(decode_video_processor) != decode_video_processor_config_seal:
                    raise ValueError("The Wan Decode video processor configuration changed during upstream execution.")
            consume_decode_route_state(
                route_state,
                binding=binding,
                model_type=getattr(self._pipeline_class, "__name__", ""),
                latents=effective_latents,
                vae_component=(
                    installed_vae if route_values["contract"] in {"sdxl", "wan_i2v"} else None
                ),
                vae_latent_channels=post_geometry[0],
                vae_scale_factor=post_geometry[1],
                video_processor=(decode_video_processor if route_values["contract"] == "wan_i2v" else None),
                execution_device=(
                    self._pipeline._execution_device if route_values["contract"] == "wan_i2v" else None
                ),
                materialize_overlay=False,
            )
            if route_values["contract"] == "wan_i2v":
                self._wan_decode_video_processor = decode_video_processor
                self._wan_decode_video_processor_config_seal = decode_video_processor_config_seal

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

    def _cache_params_equal(self, previous, current):
        equal = route_cache_params_equal(previous, current, fallback=super()._cache_params_equal)
        if not equal or not isinstance(current, dict) or self._pipeline_class is None:
            return equal
        model_type = getattr(self._pipeline_class, "__name__", "")
        route_contract = route_contract_for_model_type(model_type)
        if route_contract == "wan_i2v":
            binding = require_component_binding(
                current.get("vae"),
                label="VAE",
                expected_model_type=model_type,
                expected_role="vae",
            )
            require_cataloged_wan_action_source(
                image=current.get("image"),
                last_image=current.get("last_image"),
                binding=binding,
            )
            _blocks, node_config = require_modiff_node_contract(
                self._pipeline_class,
                self.node_type,
                resolve_blocks=False,
            )
            current = normalize_modular_runtime_params(dict(current), node_config)
            route_input = current.get(ROUTE_STATE_INPUT)
            route_output = self.output.get(ROUTE_STATE_OUTPUT)
            if route_input is None or route_output is None:
                return False
            resident_vae = require_exact_resident_component(
                getattr(self, "_pipeline", None),
                current.get("vae"),
                "vae",
                label="Encode VAE",
            )
            video_processor = getattr(getattr(self, "_pipeline", None), "video_processor", None)
            require_wan_video_processor(video_processor)
            preflight_wan_vae_route_state(
                route_input,
                binding=binding,
                model_type=model_type,
                image=current.get("image"),
                last_image=current.get("last_image"),
                height=current.get("height"),
                width=current.get("width"),
                num_frames=current.get("num_frames"),
                vae_component=resident_vae,
            )
            validate_wan_post_vae_route_state(
                route_output,
                binding=binding,
                model_type=model_type,
                seed=normalize_modular_seed(current.get("seed")),
                image_condition_latents=self.output.get("image_condition_latents"),
                height=current.get("height"),
                width=current.get("width"),
                num_frames=current.get("num_frames"),
                vae_component=resident_vae,
                video_processor=video_processor,
                producer_execution_device=getattr(self._pipeline, "_execution_device", None),
            )
            return True
        if route_contract != "sdxl":
            return True
        binding = require_component_binding(
            current.get("vae"),
            label="VAE",
            expected_model_type=model_type,
            expected_role="vae",
        )
        resident_vae = require_exact_resident_component(
            getattr(self, "_pipeline", None),
            current.get("vae"),
            "vae",
            label="Encode VAE",
        )
        route_state = self.output.get(ROUTE_STATE_OUTPUT)
        if route_state is None:
            return False
        resident_geometry = sdxl_vae_geometry_from_component(resident_vae)
        validate_encoder_route_state(
            route_state,
            binding=binding,
            model_type=model_type,
            seed=normalize_modular_seed(current.get("seed")),
            image_latents=self.output.get("image_latents"),
            mask=self.output.get("mask"),
            masked_image_latents=self.output.get("masked_image_latents"),
            vae_component=resident_vae,
            vae_latent_channels=resident_geometry[0],
            vae_scale_factor=resident_geometry[1],
        )
        return True

    def update_node(self, values, ref):
        node_params = {}
        model_type = self.get_signal_value("vae")

        if self._model_type == model_type:
            if not model_type or self._pipeline_class is None:
                return None
            _, node_config = require_modiff_node_contract(
                self._pipeline_class,
                self.node_type,
                resolve_blocks=False,
            )
            node_params_to_update = dict(node_config["params"])
            node_params_to_update.pop("vae", None)
            self.send_node_definition(node_params_to_update)
            return None

        if model_type is None or model_type == "":
            self._model_type = ""
            self._pipeline_class = None
            self.send_node_definition(node_params)
            return None
        try:
            self._pipeline_class = pipeline_class_from_model_type(model_type)
            _, node_config = require_modiff_node_contract(
                self._pipeline_class,
                self.node_type,
                resolve_blocks=False,
            )
        except ValueError:
            self._model_type = ""
            self._pipeline_class = None
            self.send_node_definition(node_params)
            raise
        self._model_type = model_type

        node_params_to_update = dict(node_config["params"])
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
        require_route_state_shape_before_identity_resolution(kwargs)
        reject_undeclared_modular_generator(kwargs)
        reject_route_reserved_inputs_before_identity_resolution(kwargs)
        self._pipeline_class = pipeline_class_from_runtime_inputs(self._pipeline_class, kwargs)
        model_type = getattr(self._pipeline_class, "__name__", "")
        route_contract = route_contract_for_model_type(model_type)
        route_binding = None
        if route_contract == "wan_i2v":
            route_binding = require_component_binding(
                kwargs.get("vae"),
                label="VAE",
                expected_model_type=model_type,
                expected_role="vae",
            )
            require_cataloged_wan_action_source(
                image=kwargs.get("image"),
                last_image=kwargs.get("last_image"),
                binding=route_binding,
            )
        # 1. Get node config
        blocks, node_config = require_modiff_node_contract(self._pipeline_class, self.node_type)
        validate_route_field_contract(kwargs, node_config)

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

        # 3. Enforce the backend-issued action schema before initializing blocks.
        kwargs = normalize_modular_runtime_params(kwargs, node_config)
        if route_contract == "sdxl":
            validate_sdxl_crop_overlay_inputs(
                kwargs.get("padding_mask_crop"),
                kwargs.get("image"),
                kwargs.get("mask_image"),
            )
        if "seed" in node_config["input_names"] and "generator" not in blocks.input_names:
            raise ValueError(
                "The backend-issued Modular Diffusers Encode Image contract declares a seed, but its installed "
                "VAE encoder block does not expose the required generator input."
            )
        preinit_vae = None
        preinit_geometry = None
        wan_preflight = None
        wan_route_values = None
        video_processor_config_seal = None
        route_state = kwargs.get(ROUTE_STATE_INPUT)
        if ROUTE_STATE_OUTPUT in node_config["output_names"]:
            if kwargs.get("seed") is None:
                raise ValueError("A Modular VAE route requires a seed before pipeline initialization.")
            if route_binding is None:
                route_binding = require_component_binding(
                    vae,
                    label="VAE",
                    expected_model_type=model_type,
                    expected_role="vae",
                )
            if route_contract == "sdxl":
                preinit_vae = resolve_managed_component_by_id(components, vae, label="Encode VAE")
                preinit_geometry = sdxl_vae_geometry_from_component(preinit_vae)
            elif route_contract == "wan_i2v":
                if route_state is None:
                    raise ValueError(
                        "Wan Image Encode requires the opaque route emitted by Image Embeddings."
                    )
                preinit_vae = resolve_managed_component_by_id(components, vae, label="Encode VAE")
                wan_preflight = preflight_wan_vae_route_state(
                    route_state,
                    binding=route_binding,
                    model_type=model_type,
                    image=kwargs.get("image"),
                    last_image=kwargs.get("last_image"),
                    height=kwargs.get("height"),
                    width=kwargs.get("width"),
                    num_frames=kwargs.get("num_frames"),
                    vae_component=preinit_vae,
                )
                wan_route_values = consume_wan_image_encoder_route_state(
                    route_state,
                    binding=route_binding,
                    model_type=model_type,
                    image=kwargs.get("image"),
                    last_image=kwargs.get("last_image"),
                    height=kwargs.get("height"),
                    width=kwargs.get("width"),
                )
        self._pipeline = blocks.init_pipeline(components_manager=components)

        # 4. Update components
        expected_component_names = blocks.component_names
        model_input_names = node_config["model_input_names"]
        model_ids = collect_model_ids(
            kwargs,
            target_key_names=model_input_names,
            target_model_names=expected_component_names,
        )

        components_to_update = {}
        if model_ids:
            components_to_update = components.get_components_by_ids(ids=model_ids, return_dict_with_names=True)
            if components_to_update:
                self._pipeline.update_components(**components_to_update)
        route_geometry = None
        if route_binding is not None:
            require_component_binding(
                vae,
                label="VAE",
                expected_model_type=model_type,
                expected_token=route_binding,
                expected_role="vae",
            )
            installed_vae = require_exact_installed_component(
                self._pipeline,
                components_to_update,
                expected_component_names,
                "vae",
            )
            live_vae = installed_vae
            if installed_vae is not None:
                live_vae = require_exact_resident_component(
                    self._pipeline,
                    vae,
                    "vae",
                    label="Encode VAE",
                )
                if live_vae is not installed_vae:
                    raise ValueError("The connected Encode VAE changed during component installation.")
            if route_contract == "sdxl":
                if live_vae is not preinit_vae:
                    raise ValueError("The connected Encode VAE changed during pipeline initialization.")
                route_geometry = sdxl_vae_geometry_from_component(live_vae)
                if route_geometry != preinit_geometry:
                    raise ValueError("The connected Encode VAE geometry changed during pipeline initialization.")
            elif route_contract == "wan_i2v":
                if live_vae is not preinit_vae:
                    raise ValueError("The connected Wan Encode VAE changed during pipeline initialization.")
                video_processor = require_wan_video_processor(getattr(self._pipeline, "video_processor", None))
                video_processor_config_seal = wan_video_processor_config_seal(video_processor)
                installed_preflight = preflight_wan_vae_route_state(
                    route_state,
                    binding=route_binding,
                    model_type=model_type,
                    image=kwargs.get("image"),
                    last_image=kwargs.get("last_image"),
                    height=kwargs.get("height"),
                    width=kwargs.get("width"),
                    num_frames=kwargs.get("num_frames"),
                    vae_component=live_vae,
                )
                if installed_preflight != wan_preflight:
                    raise ValueError("Wan VAE geometry changed during pipeline initialization.")

        # 5. Compile runtime inputs from kwargs based on node_config["input_names"]
        node_kwargs = {}
        encoder_generator = None
        input_names = node_config["input_names"]

        for name in input_names:
            if name not in kwargs:
                continue
            value = kwargs.get(name)

            if name == "seed":
                if value is not None:
                    encoder_generator = modular_generator_from_seed(value, self._pipeline)
                    node_kwargs["generator"] = encoder_generator
            elif isinstance(value, dict) and name not in blocks.input_names:
                for k, v in value.items():
                    if k in blocks.input_names:
                        node_kwargs[k] = v
            elif name in blocks.input_names:
                node_kwargs[name] = value

        if wan_route_values is not None:
            # Preserve the raw requested values only in kwargs/cache identity.
            # The second upstream resize must receive the first-pass geometry.
            node_kwargs["height"] = wan_route_values["height"]
            node_kwargs["width"] = wan_route_values["width"]

        if "image" in node_kwargs:
            node_kwargs["image"] = prepare_image_for_vae_pipeline(node_kwargs["image"], self._pipeline_class)

        # 6. Run the pipeline
        if route_contract == "wan_i2v":
            if require_exact_resident_component(
                self._pipeline,
                vae,
                "vae",
                label="Encode VAE",
            ) is not installed_vae:
                raise ValueError("The connected Wan Encode VAE changed before upstream execution.")
            if getattr(self._pipeline, "video_processor", None) is not video_processor:
                raise ValueError("The Wan VAE video processor changed before upstream execution.")
            if wan_video_processor_config_seal(video_processor) != video_processor_config_seal:
                raise ValueError("The Wan VAE video processor configuration changed before upstream execution.")
            if preflight_wan_vae_route_state(
                route_state,
                binding=route_binding,
                model_type=model_type,
                image=kwargs.get("image"),
                last_image=kwargs.get("last_image"),
                height=kwargs.get("height"),
                width=kwargs.get("width"),
                num_frames=kwargs.get("num_frames"),
                vae_component=installed_vae,
            ) != wan_preflight:
                raise ValueError("Wan VAE geometry changed before upstream execution.")
        try:
            node_output_state = self._pipeline(**node_kwargs)
        except ValueError as e:
            self.notify(str(e), variant="error", persist=False, autoHideDuration=MESSAGE_DURATION)
            raise
        if route_binding is not None:
            require_component_binding(
                vae,
                label="VAE",
                expected_model_type=model_type,
                expected_token=route_binding,
                expected_role="vae",
            )
            installed_vae = require_exact_installed_component(
                self._pipeline,
                components_to_update,
                expected_component_names,
                "vae",
            )
            live_vae = installed_vae
            if installed_vae is not None:
                live_vae = require_exact_resident_component(
                    self._pipeline,
                    vae,
                    "vae",
                    label="Encode VAE",
                )
                if live_vae is not installed_vae:
                    raise ValueError("The connected Encode VAE changed during upstream execution.")
            if route_geometry is not None and sdxl_vae_geometry_from_component(installed_vae) != route_geometry:
                raise ValueError("The connected Encode VAE geometry changed during upstream execution.")
            if route_contract == "wan_i2v":
                if live_vae is not preinit_vae:
                    raise ValueError("The connected Wan Encode VAE changed during upstream execution.")
                if getattr(self._pipeline, "video_processor", None) is not video_processor:
                    raise ValueError("The Wan VAE video processor changed during upstream execution.")
                if wan_video_processor_config_seal(video_processor) != video_processor_config_seal:
                    raise ValueError("The Wan VAE video processor configuration changed during upstream execution.")
                if preflight_wan_vae_route_state(
                    route_state,
                    binding=route_binding,
                    model_type=model_type,
                    image=kwargs.get("image"),
                    last_image=kwargs.get("last_image"),
                    height=kwargs.get("height"),
                    width=kwargs.get("width"),
                    num_frames=kwargs.get("num_frames"),
                    vae_component=live_vae,
                ) != wan_preflight:
                    raise ValueError("Wan VAE geometry changed during upstream execution.")

        # 7. Prepare outputs based on node_config["output_names"]
        output_names = node_config["output_names"].copy()
        outputs = {}
        for name in output_names:
            if name == "doc":
                outputs["doc"] = self._pipeline.blocks.doc
            elif name == ROUTE_STATE_OUTPUT:
                if encoder_generator is None or route_binding is None:
                    raise ValueError("The Modular VAE route is missing its validated generator or loader binding.")
                if route_contract == "wan_i2v":
                    raw_name = "first_last_frame_latents" if wan_preflight[0] == "flf2v" else "first_frame_latents"
                    outputs[name] = issue_wan_vae_route_state(
                        route_state,
                        binding=route_binding,
                        seed=kwargs["seed"],
                        generator=encoder_generator,
                        image=kwargs.get("image"),
                        last_image=kwargs.get("last_image"),
                        height=kwargs.get("height"),
                        width=kwargs.get("width"),
                        num_frames=kwargs.get("num_frames"),
                        image_condition_latents=node_output_state.get("image_condition_latents"),
                        raw_frame_latents=node_output_state.get(raw_name),
                        vae_component=installed_vae,
                        video_processor=video_processor,
                        resized_image=node_output_state.get("resized_image"),
                        resized_last_image=node_output_state.get("resized_last_image"),
                        execution_device=self._pipeline._execution_device,
                        preflight_geometry=wan_preflight,
                    )
                    continue
                route_kwargs = {
                    "binding": route_binding,
                    "seed": kwargs["seed"],
                    "generator": encoder_generator,
                    "image_latents": node_output_state.get("image_latents"),
                    "processed_mask_image": node_output_state.get("processed_mask_image"),
                    "mask_overlay_kwargs": node_output_state.get("mask_overlay_kwargs"),
                }
                if route_contract == "sdxl":
                    route_kwargs.update(
                        mask=node_output_state.get("mask"),
                        masked_image_latents=node_output_state.get("masked_image_latents"),
                        padding_mask_crop=kwargs.get("padding_mask_crop"),
                        crops_coords=node_output_state.get("crops_coords"),
                        original_image=node_kwargs.get("image"),
                        original_mask=node_kwargs.get("mask_image"),
                        vae_component=installed_vae,
                        vae_latent_channels=route_geometry[0],
                        vae_scale_factor=route_geometry[1],
                    )
                outputs[name] = issue_encoder_route_state(
                    **route_kwargs,
                )
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
