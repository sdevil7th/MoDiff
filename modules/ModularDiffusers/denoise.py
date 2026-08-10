# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import inspect
import logging
import time
from copy import deepcopy
from typing import Any, List, Tuple

import torch
from diffusers import BaseGuidance, ComponentsManager
from diffusers.modular_pipelines import BlockState, LoopSequentialPipelineBlocks, ModularPipelineBlocks

from modiff.NodeBase import NodeBase

from . import MESSAGE_DURATION, components
from .modular_utils import (
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
    consume_decode_route_state,
    SUPPORTED_ROUTE_MODEL_TYPES,
    consume_denoise_route_state,
    effective_modular_block_input,
    issue_decode_route_state,
    issue_normal_decode_route_state,
    reject_route_reserved_inputs,
    reject_route_reserved_inputs_before_identity_resolution,
    resolve_managed_component_by_id,
    require_component_binding,
    require_matching_token_bearers,
    require_route_state_shape_before_identity_resolution,
    route_contract_for_model_type,
    route_requires_controlnet_state,
    route_cache_params_equal,
    route_uses_hidden_denoise_mask,
    sdxl_vae_geometry_from_component,
    validate_denoise_route_state,
    validate_route_field_contract,
    wan_transformer_contract_from_component,
    wan_vae_geometry_from_component,
)
from .utils import collect_model_ids


logger = logging.getLogger("modiff")
_MISSING_SIGNATURE = object()

MISSING_EMBEDDINGS_MESSAGE = (
    "Prompt embeddings are missing from Encode Prompt. "
    "Update or recreate the Studio graph after the model fields finish refreshing."
)


def embeddings_missing_error(error):
    return bool(error.args and error.args[0] == "embeddings") or "embeddings" in str(error)


def embeddings_are_missing(embeddings):
    return embeddings is None or (isinstance(embeddings, dict) and not embeddings)


def restore_wrapped_forward_signature(model):
    """Expose canonical model kwargs while Diffusers hooks wrap ``forward``.

    Modular denoisers inspect the runtime forward signature to decide which
    conditioning fields to pass. Group-offload/quantization hooks may replace
    it with ``(*args, **kwargs)``, which silently drops fields such as Qwen
    Layered's ``additional_t_cond``.
    """
    runtime_forward = getattr(model, "forward", None)
    class_forward = getattr(type(model), "forward", None)
    if runtime_forward is None or class_forward is None:
        return None

    runtime_params = inspect.signature(runtime_forward).parameters
    canonical_signature = inspect.signature(class_forward)
    canonical_params = canonical_signature.parameters
    if set(canonical_params).issubset(runtime_params):
        return None

    signature_target = getattr(runtime_forward, "__func__", runtime_forward)
    previous = getattr(signature_target, "__signature__", _MISSING_SIGNATURE)
    signature_target.__signature__ = canonical_signature
    logger.debug(
        "Restored wrapped %s.forward signature for modular conditioning fields: %s",
        type(model).__name__,
        sorted(set(canonical_params) - set(runtime_params)),
    )
    return signature_target, previous


def reset_wrapped_forward_signature(signature_state):
    if signature_state is None:
        return
    signature_target, previous = signature_state
    if previous is _MISSING_SIGNATURE:
        try:
            del signature_target.__signature__
        except AttributeError:
            pass
    else:
        signature_target.__signature__ = previous


class PreviewBlock(ModularPipelineBlocks):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return []

    def __call__(self, components: ComponentsManager, block_state: BlockState, i: int, t: int):
        self.callback(block_state.latents, i, components.scheduler.order)

        return components, block_state


def insert_preview_block(blocks, callback):
    """Insert preview_block into all LoopSequentialPipelineBlocks with 'denoise' in name."""

    # new block so we can preview the generation
    preview_block = PreviewBlock(callback)

    def insert_preview_block_recursive(blocks, blocks_name, preview_block):
        if hasattr(blocks, "sub_blocks"):
            if isinstance(blocks, LoopSequentialPipelineBlocks) and "denoise" in blocks_name.lower():
                blocks.sub_blocks.insert("preview_block", preview_block, len(blocks.sub_blocks))
            else:
                for sub_block_name, sub_block in blocks.sub_blocks.items():
                    insert_preview_block_recursive(sub_block, sub_block_name, preview_block)

    insert_preview_block_recursive(blocks, "root", preview_block)


class Denoise(NodeBase):
    label = "Denoise"
    category = "sampler"
    resizable = True
    skipParamsCheck = True
    node_type = "denoise"
    params = {
        "unet": {
            "label": "Denoise Model *",
            "display": "input",
            "type": "diffusers_auto_model",
            "required": True,
            "onSignal": [
                "update_node",
                {"action": "signal", "target": "guider"},
                {"action": "signal", "target": "controlnet_bundle"},
            ],
        },
    }

    def update_node(self, values, ref):
        node_params = {}
        model_type = self.get_signal_value("unet")

        if self._model_type == model_type:
            if not model_type or self._pipeline_class is None:
                return None
            _, node_config = require_modiff_node_contract(
                self._pipeline_class,
                self.node_type,
                resolve_blocks=False,
            )
            node_params_to_update = dict(node_config["params"])
            node_params_to_update.pop("unet", None)
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
        node_params_to_update.pop("unet", None)

        node_params.update(**node_params_to_update)
        self.send_node_definition(node_params)

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._model_type = ""
        self._pipeline_class = None
        self._route_cache_node_input_names = ()
        self._route_cache_block_input_names = ()
        self._route_cache_component_names = ()
        self._route_cache_model_input_names = ()

    def _require_route_component_inputs(self, current, *, model_input_names, bundle_names):
        binding = require_component_binding(
            current.get("unet"),
            label="denoise model",
            expected_model_type=self._model_type,
            expected_role="denoiser",
        )
        require_component_binding(
            current.get("scheduler"),
            label="scheduler",
            expected_model_type=self._model_type,
            expected_token=binding,
            expected_role="scheduler",
        )
        if "vae" in model_input_names:
            require_component_binding(
                current.get("vae"),
                label="Denoise VAE",
                expected_model_type=self._model_type,
                expected_token=binding,
                expected_role="vae",
            )
        for bundle_name in bundle_names:
            require_matching_token_bearers(current[bundle_name], binding, label=bundle_name)
        return binding

    def _require_installed_route_components(self, current, component_updates, *, model_input_names):
        """Prove required connected components were installed, not ambient manager picks."""

        if "vae" not in model_input_names:
            return None
        if route_contract_for_model_type(self._model_type) == "wan_i2v":
            vae = resolve_managed_component_by_id(components, current.get("vae"), label="Denoise VAE")
            wan_vae_geometry_from_component(vae)
            return vae
        expected_vae = component_updates.get("vae")
        if expected_vae is None:
            raise ValueError("The connected Denoise VAE could not be resolved from its exact managed component ID.")
        if getattr(self._pipeline, "vae", None) is not expected_vae:
            raise ValueError("The SDXL Denoise pipeline did not install the exact connected VAE component.")
        return expected_vae

    def _resolve_wan_route_components(self, current, *, require_resident_pipeline):
        vae = resolve_managed_component_by_id(components, current.get("vae"), label="Denoise VAE")
        transformer = resolve_managed_component_by_id(
            components,
            current.get("unet"),
            label="Denoise transformer",
        )
        wan_vae_geometry_from_component(vae)
        wan_transformer_contract_from_component(transformer)
        if require_resident_pipeline and (
            getattr(self, "_pipeline", None) is None
            or getattr(self._pipeline, "transformer", None) is not transformer
        ):
            raise ValueError("The resident Wan Denoise pipeline does not hold the exact connected transformer.")
        return vae, transformer

    def _resolve_sdxl_route_vae(self, current, *, require_resident_pipeline):
        vae = resolve_managed_component_by_id(components, current.get("vae"), label="Denoise VAE")
        if require_resident_pipeline and (
            getattr(self, "_pipeline", None) is None or getattr(self._pipeline, "vae", None) is not vae
        ):
            raise ValueError("The resident SDXL Denoise pipeline does not hold the exact connected VAE component.")
        return vae, sdxl_vae_geometry_from_component(vae)

    def _validate_route_cache_inputs(self, current):
        route_state = current.get(ROUTE_STATE_INPUT)
        supported_route_model = self._model_type in SUPPORTED_ROUTE_MODEL_TYPES
        if route_state is None and not supported_route_model:
            return True
        if not self._route_cache_node_input_names or not self._route_cache_block_input_names:
            return False

        route_contract = route_contract_for_model_type(self._model_type)
        if route_contract == "wan_i2v":
            _blocks, node_config = require_modiff_node_contract(
                self._pipeline_class,
                self.node_type,
                resolve_blocks=False,
            )
            current = normalize_modular_runtime_params(dict(current), node_config)
            route_state = current.get(ROUTE_STATE_INPUT)

        block_input_names = self._route_cache_block_input_names
        node_input_names = self._route_cache_node_input_names
        component_names = self._route_cache_component_names
        bundle_names = [
            name
            for name in node_input_names
            if isinstance(current.get(name), dict) and name not in block_input_names
        ]
        if route_contract == "sdxl":
            if current.get("ip_adapter") is not None:
                raise ValueError("SDXL IP-Adapter execution is not enabled.")
            for bundle_name in bundle_names:
                ip_fields = {"ip_adapter_embeds", "negative_ip_adapter_embeds"}.intersection(
                    current[bundle_name]
                )
                if ip_fields:
                    raise ValueError(
                        f"SDXL IP-Adapter fields are not enabled: {', '.join(sorted(ip_fields))}."
                    )
                union_fields = {"control_mode", "control_type", "control_type_idx"}.intersection(
                    current[bundle_name]
                )
                if union_fields:
                    raise ValueError(
                        f"SDXL ControlNet Union fields are not enabled: {', '.join(sorted(union_fields))}."
                    )
        require_route_state_shape_before_identity_resolution(current)
        reject_undeclared_modular_generator(current)
        reject_route_reserved_inputs_before_identity_resolution(
            current,
            allowed_direct_inputs={"mask", "masked_image_latents"},
        )
        reject_route_reserved_inputs(
            current,
            bundle_names=bundle_names,
            model_type=self._model_type,
            action="denoise",
        )
        binding = self._require_route_component_inputs(
            current,
            model_input_names=self._route_cache_model_input_names,
            bundle_names=bundle_names,
        )
        resident_vae = None
        resident_geometry = (None, None)
        resident_transformer = None
        if route_contract == "sdxl":
            resident_vae, resident_geometry = self._resolve_sdxl_route_vae(
                current,
                require_resident_pipeline=True,
            )
            cached_decode_route = self.output.get(ROUTE_STATE_OUTPUT)
            if cached_decode_route is None:
                return False
            consume_decode_route_state(
                cached_decode_route,
                binding=binding,
                model_type=self._model_type,
                latents=self.output.get("latents"),
                vae_component=resident_vae,
                vae_latent_channels=resident_geometry[0],
                vae_scale_factor=resident_geometry[1],
                materialize_overlay=False,
            )
        elif route_contract == "wan_i2v":
            resident_vae, resident_transformer = self._resolve_wan_route_components(
                current,
                require_resident_pipeline=True,
            )
            cached_decode_route = self.output.get(ROUTE_STATE_OUTPUT)
            if cached_decode_route is None:
                return False
            consume_decode_route_state(
                cached_decode_route,
                binding=binding,
                model_type=self._model_type,
                latents=self.output.get("latents"),
                vae_component=resident_vae,
                materialize_overlay=False,
            )

        image_latents = effective_modular_block_input(
            current,
            node_input_names=node_input_names,
            block_input_names=block_input_names,
            target_name="image_latents",
        )
        mask = effective_modular_block_input(
            current,
            node_input_names=node_input_names,
            block_input_names=block_input_names,
            target_name="mask",
        )
        masked_image_latents = effective_modular_block_input(
            current,
            node_input_names=node_input_names,
            block_input_names=block_input_names,
            target_name="masked_image_latents",
        )
        control_image_latents = effective_modular_block_input(
            current,
            node_input_names=node_input_names,
            block_input_names=block_input_names,
            target_name="control_image_latents",
        )
        controlnet_component = effective_modular_block_input(
            current,
            node_input_names=node_input_names,
            block_input_names=tuple(block_input_names) + tuple(component_names),
            target_name="controlnet",
        )
        controlnet_bundle_present = current.get("controlnet_bundle") is not None
        image_embeds = effective_modular_block_input(
            current,
            node_input_names=node_input_names,
            block_input_names=block_input_names,
            target_name="image_embeds",
        )
        image_condition_latents = effective_modular_block_input(
            current,
            node_input_names=node_input_names,
            block_input_names=block_input_names,
            target_name="image_condition_latents",
        )
        if route_state is None:
            if route_contract == "wan_i2v":
                raise ValueError("Wan image-to-video Denoise requires its preceding Image Encode route.")
            if image_latents is not None or mask is not None or masked_image_latents is not None or any(
                current.get(name) is not None for name in ("image_latents", "image_latents_with_strength")
            ):
                raise ValueError(
                    "Connected Modular image or inpaint latents require the opaque route state emitted by the "
                    "matching VAE encoder."
                )
            if route_requires_controlnet_state(self._model_type) and (
                controlnet_bundle_present
                or control_image_latents is not None
                or controlnet_component is not None
            ):
                raise ValueError(
                    "Connected Qwen ControlNet inputs require the opaque route state emitted by the matching "
                    "ControlNet action."
                )
            return True
        validate_denoise_route_state(
            route_state,
            binding=binding,
            model_type=self._model_type,
            seed=normalize_modular_seed(current.get("seed")),
            image_latents=image_latents,
            mask=mask,
            masked_image_latents=masked_image_latents,
            vae_component=resident_vae,
            vae_latent_channels=resident_geometry[0],
            vae_scale_factor=resident_geometry[1],
            control_image_latents=control_image_latents,
            controlnet_component=controlnet_component,
            controlnet_bundle_present=controlnet_bundle_present,
            ip_adapter_present=current.get("ip_adapter") is not None,
            image_embeds=image_embeds,
            image_condition_latents=image_condition_latents,
            height=current.get("height"),
            width=current.get("width"),
            num_frames=current.get("num_frames"),
            transformer_component=resident_transformer,
        )
        return True

    def _cache_params_equal(self, previous, current):
        equal = route_cache_params_equal(previous, current, fallback=super()._cache_params_equal)
        if equal and isinstance(current, dict) and not self._validate_route_cache_inputs(current):
            return False
        return equal

    def _raise_if_interrupted(self):
        if self._interrupt:
            raise InterruptedError("Execution interrupted by the user.")

    def _publish_initial_denoise_progress(self, num_inference_steps: int):
        if num_inference_steps <= 0:
            return
        self.progress(
            0,
            phase="denoising",
            message=f"Denoising 0/{num_inference_steps}",
            current_step=0,
            total_steps=num_inference_steps,
            elapsed_seconds=0.0,
            average_step_seconds=None,
            eta_seconds=None,
        )

    def execute(self, **kwargs):
        kwargs = dict(kwargs)
        require_route_state_shape_before_identity_resolution(kwargs)
        reject_undeclared_modular_generator(kwargs)
        reject_route_reserved_inputs_before_identity_resolution(
            kwargs,
            allowed_direct_inputs={"mask", "masked_image_latents"},
        )
        identity_kwargs = {name: value for name, value in kwargs.items() if name != ROUTE_STATE_INPUT}
        self._pipeline_class = pipeline_class_from_runtime_inputs(self._pipeline_class, identity_kwargs)
        self._model_type = getattr(self._pipeline_class, "__name__", "")

        if not ((unet := kwargs.get("unet")) and isinstance(unet, dict)):
            self.notify(
                "You have to connect the denoise model",
                variant="error",
                persist=False,
                autoHideDuration=MESSAGE_DURATION,
            )
            return None

        # 1. Get node config
        blocks, node_config = require_modiff_node_contract(self._pipeline_class, self.node_type)
        validate_route_field_contract(kwargs, node_config)

        route_state = kwargs.get(ROUTE_STATE_INPUT)
        route_binding = None
        supported_route_model = self._model_type in SUPPORTED_ROUTE_MODEL_TYPES
        route_contract = route_contract_for_model_type(self._model_type)
        model_input_names = node_config["model_input_names"]
        bundle_names = [
            name
            for name in node_config["input_names"]
            if isinstance(kwargs.get(name), dict) and name not in blocks.input_names
        ]
        if route_contract == "sdxl":
            if kwargs.get("ip_adapter") is not None:
                raise ValueError("SDXL IP-Adapter execution is not enabled.")
            for bundle_name in bundle_names:
                ip_fields = {"ip_adapter_embeds", "negative_ip_adapter_embeds"}.intersection(
                    kwargs[bundle_name]
                )
                if ip_fields:
                    raise ValueError(
                        f"SDXL IP-Adapter fields are not enabled: {', '.join(sorted(ip_fields))}."
                    )
                union_fields = {"control_mode", "control_type", "control_type_idx"}.intersection(
                    kwargs[bundle_name]
                )
                if union_fields:
                    raise ValueError(
                        f"SDXL ControlNet Union fields are not enabled: {', '.join(sorted(union_fields))}."
                    )
        effective_image_latents = effective_modular_block_input(
            kwargs,
            node_input_names=node_config["input_names"],
            block_input_names=blocks.input_names,
            target_name="image_latents",
        )
        effective_strength_latents = effective_modular_block_input(
            kwargs,
            node_input_names=node_config["input_names"],
            block_input_names=blocks.input_names,
            target_name="image_latents_with_strength",
        )
        effective_mask = effective_modular_block_input(
            kwargs,
            node_input_names=node_config["input_names"],
            block_input_names=blocks.input_names,
            target_name="mask",
        )
        effective_masked_image_latents = effective_modular_block_input(
            kwargs,
            node_input_names=node_config["input_names"],
            block_input_names=blocks.input_names,
            target_name="masked_image_latents",
        )
        effective_control_latents = effective_modular_block_input(
            kwargs,
            node_input_names=node_config["input_names"],
            block_input_names=blocks.input_names,
            target_name="control_image_latents",
        )
        effective_image_embeds = effective_modular_block_input(
            kwargs,
            node_input_names=node_config["input_names"],
            block_input_names=blocks.input_names,
            target_name="image_embeds",
        )
        effective_image_condition_latents = effective_modular_block_input(
            kwargs,
            node_input_names=node_config["input_names"],
            block_input_names=blocks.input_names,
            target_name="image_condition_latents",
        )
        component_names = getattr(blocks, "component_names", ())
        if not isinstance(component_names, (list, tuple, set, frozenset)):
            component_names = ()
        self._route_cache_node_input_names = tuple(node_config["input_names"])
        self._route_cache_block_input_names = tuple(blocks.input_names)
        self._route_cache_component_names = tuple(component_names)
        self._route_cache_model_input_names = tuple(node_config["model_input_names"])
        effective_controlnet_component = effective_modular_block_input(
            kwargs,
            node_input_names=node_config["input_names"],
            block_input_names=tuple(blocks.input_names) + tuple(component_names),
            target_name="controlnet",
        )
        controlnet_bundle_present = kwargs.get("controlnet_bundle") is not None
        routed_latent_present = any(
            kwargs.get(name) is not None for name in ("image_latents", "image_latents_with_strength")
        )
        routed_latent_present = routed_latent_present or any(
            value is not None
            for value in (
                effective_image_latents,
                effective_strength_latents,
                effective_mask,
                effective_masked_image_latents,
            )
        )
        if (
            route_state is None
            and supported_route_model
            and routed_latent_present
        ):
            raise ValueError(
                "Connected Modular image or inpaint latents require the opaque route state emitted by the matching "
                "VAE encoder."
            )
        if route_state is None and route_contract == "wan_i2v":
            raise ValueError("Wan image-to-video Denoise requires its preceding Image Encode route.")
        controlnet_state_present = (
            controlnet_bundle_present
            or effective_control_latents is not None
            or effective_controlnet_component is not None
        )
        if route_state is None and route_requires_controlnet_state(self._model_type) and controlnet_state_present:
            raise ValueError(
                "Connected Qwen ControlNet inputs require the opaque route state emitted by the matching "
                "ControlNet action."
            )
        if route_state is not None:
            # Route-enabled actions use the strict backend schema before any
            # block deepcopy or pipeline initialization. In particular, do not
            # let the compatibility int() cast normalize bools or fractions.
            kwargs = normalize_modular_runtime_params(kwargs, node_config)
        if route_state is not None and not supported_route_model:
            raise ValueError(f"Pipeline '{self._model_type}' does not support opaque Modular route state.")

        if supported_route_model:
            reject_route_reserved_inputs(
                kwargs,
                bundle_names=bundle_names,
                model_type=self._model_type,
                action="denoise",
            )
            route_binding = self._require_route_component_inputs(
                kwargs,
                model_input_names=node_config["model_input_names"],
                bundle_names=bundle_names,
            )
        preinit_vae = None
        preinit_geometry = (None, None)
        preinit_transformer = None
        if route_contract == "sdxl":
            preinit_vae, preinit_geometry = self._resolve_sdxl_route_vae(
                kwargs,
                require_resident_pipeline=False,
            )
        elif route_contract == "wan_i2v":
            preinit_vae, preinit_transformer = self._resolve_wan_route_components(
                kwargs,
                require_resident_pipeline=False,
            )

        if route_state is not None:
            if kwargs.get("seed") is None:
                raise ValueError("A Modular VAE-to-Denoise route requires its originating seed.")
            validate_denoise_route_state(
                route_state,
                binding=route_binding,
                model_type=self._model_type,
                seed=kwargs["seed"],
                image_latents=effective_image_latents,
                mask=effective_mask,
                masked_image_latents=effective_masked_image_latents,
                vae_component=preinit_vae,
                vae_latent_channels=preinit_geometry[0],
                vae_scale_factor=preinit_geometry[1],
                control_image_latents=effective_control_latents,
                controlnet_component=effective_controlnet_component,
                controlnet_bundle_present=controlnet_bundle_present,
                ip_adapter_present=kwargs.get("ip_adapter") is not None,
                image_embeds=effective_image_embeds,
                image_condition_latents=effective_image_condition_latents,
                height=kwargs.get("height"),
                width=kwargs.get("width"),
                num_frames=kwargs.get("num_frames"),
                transformer_component=preinit_transformer,
            )

        if "embeddings" in node_config["input_names"]:
            embeddings = kwargs.get("embeddings")
            if embeddings_are_missing(embeddings):
                self.notify(
                    MISSING_EMBEDDINGS_MESSAGE,
                    variant="error",
                    persist=False,
                    autoHideDuration=MESSAGE_DURATION,
                )
                return None

        # 2. create pipeline
        num_inference_steps = int(kwargs.get("num_inference_steps") or 0)
        progress_started_at = time.monotonic()

        def preview_callback(_latents, step_index: int, scheduler_order: int):
            # Modular pipelines do not expose the standard Diffusers
            # ``callback_on_step_end`` contract used by NodeBase.pipe_callback.
            # This block runs at the denoise step boundary, so it is the safe
            # place to honor an app stop request without interrupting a GPU
            # kernel and poisoning the accelerator context.
            self._raise_if_interrupted()

            if num_inference_steps <= 0 or (step_index + 1) % scheduler_order != 0:
                return
            current_step = min(num_inference_steps, (step_index + 1) // scheduler_order)
            progress = int(current_step / num_inference_steps * 100)
            elapsed_seconds = max(0.0, time.monotonic() - progress_started_at)
            average_step_seconds = elapsed_seconds / current_step
            eta_seconds = average_step_seconds * max(0, num_inference_steps - current_step)
            self.progress(
                progress,
                phase="denoising",
                message=f"Denoising {current_step}/{num_inference_steps}",
                current_step=current_step,
                total_steps=num_inference_steps,
                elapsed_seconds=elapsed_seconds,
                average_step_seconds=average_step_seconds,
                eta_seconds=eta_seconds,
            )

        runtime_blocks = deepcopy(blocks)
        insert_preview_block(runtime_blocks, preview_callback)
        # The selected installed blocks and connected managed components are
        # already the reviewed execution contract. Passing the repository here
        # would make upstream reload its mutable config and resolve type hints a
        # second time, outside ModelsLoader's exact-index validation.
        self._pipeline = runtime_blocks.init_pipeline(components_manager=components)
        if supported_route_model:
            self._require_route_component_inputs(
                kwargs,
                model_input_names=model_input_names,
                bundle_names=bundle_names,
            )

        # Preserve the graph compatibility cast until the upstream schema exposes exact types.
        for param_name, param_config in node_config["params"].items():
            if param_name in kwargs and kwargs[param_name] is not None:
                param_type = param_config.get("type", None)
                if param_type == "float":
                    kwargs[param_name] = float(kwargs[param_name])
                elif param_type == "int":
                    kwargs[param_name] = int(kwargs[param_name])

        # 3. update components
        expected_component_names = blocks.component_names
        model_input_names = node_config["model_input_names"]
        install_model_input_names = model_input_names
        if route_contract == "wan_i2v":
            # Validation/provenance-only: pinned split Wan Denoise has no VAE
            # component slot, so never inject this graph port upstream.
            install_model_input_names = [name for name in model_input_names if name != "vae"]
        model_ids = collect_model_ids(
            kwargs,
            target_key_names=install_model_input_names,
            target_model_names=expected_component_names,
        )

        component_updates = {}
        explicit_guider = kwargs.get("guider")
        if explicit_guider is not None:
            if not isinstance(explicit_guider, BaseGuidance):
                guider_type = f"{type(explicit_guider).__module__}.{type(explicit_guider).__qualname__}"
                raise TypeError(
                    "Connected guider must be a Diffusers BaseGuidance instance; "
                    f"received {guider_type}."
                )
            if "guider" not in self._pipeline.component_names:
                raise ValueError(
                    f"{type(self._pipeline).__name__} does not expose a 'guider' component, "
                    "so the connected Diffusers guider cannot be installed."
                )

        if model_ids:
            managed_components = components.get_components_by_ids(ids=model_ids, return_dict_with_names=True)
            if managed_components:
                component_updates.update(managed_components)

        if explicit_guider is not None:
            component_updates["guider"] = explicit_guider

        if component_updates:
            self._pipeline.update_components(**component_updates)
        route_geometry = None
        route_transformer = None
        if supported_route_model:
            self._require_route_component_inputs(
                kwargs,
                model_input_names=model_input_names,
                bundle_names=bundle_names,
            )
            installed_vae = self._require_installed_route_components(
                kwargs,
                component_updates,
                model_input_names=model_input_names,
            )
            if route_contract == "sdxl":
                live_vae, live_geometry = self._resolve_sdxl_route_vae(
                    kwargs,
                    require_resident_pipeline=True,
                )
                if live_vae is not installed_vae:
                    raise ValueError("The connected Denoise VAE changed during component installation.")
                if installed_vae is not preinit_vae:
                    raise ValueError("The connected Denoise VAE changed during pipeline initialization.")
                route_geometry = live_geometry
                if route_geometry != preinit_geometry:
                    raise ValueError("The connected Denoise VAE geometry changed during pipeline initialization.")
            elif route_contract == "wan_i2v":
                live_vae, route_transformer = self._resolve_wan_route_components(
                    kwargs,
                    require_resident_pipeline=True,
                )
                if live_vae is not installed_vae or live_vae is not preinit_vae:
                    raise ValueError("The connected Wan Denoise VAE changed during pipeline initialization.")
                if route_transformer is not preinit_transformer:
                    raise ValueError("The connected Wan transformer changed during pipeline initialization.")

        device = self._pipeline._execution_device
        route_runtime_inputs = None
        if route_state is not None:
            route_runtime_inputs = consume_denoise_route_state(
                route_state,
                binding=route_binding,
                model_type=self._model_type,
                seed=kwargs["seed"],
                execution_device=device,
                image_latents=effective_image_latents,
                mask=effective_mask,
                masked_image_latents=effective_masked_image_latents,
                vae_component=installed_vae,
                vae_latent_channels=(route_geometry[0] if route_geometry is not None else None),
                vae_scale_factor=(route_geometry[1] if route_geometry is not None else None),
                control_image_latents=effective_control_latents,
                controlnet_component=effective_controlnet_component,
                controlnet_bundle_present=controlnet_bundle_present,
                ip_adapter_present=kwargs.get("ip_adapter") is not None,
                image_embeds=effective_image_embeds,
                image_condition_latents=effective_image_condition_latents,
                height=kwargs.get("height"),
                width=kwargs.get("width"),
                num_frames=kwargs.get("num_frames"),
                transformer_component=route_transformer,
            )

        # 4. compile a dict of runtime inputs from kwargs based on node_config["input_names"]
        node_kwargs = {}
        input_names = node_config["input_names"]

        for name in input_names:
            value = kwargs.get(name)
            if value is None:
                continue

            if name == ROUTE_STATE_INPUT:
                continue

            # special case #1: `seed` -> always create a `generator`
            if name == "seed":
                if route_runtime_inputs is None:
                    generator = torch.Generator(device=device).manual_seed(value)
                    node_kwargs["generator"] = generator

            # special case #2: passed `guidance_scale` but pipeline does not accept it
            # -> potentially create a new guider if pipeline support it
            elif name == "guidance_scale" and "guidance_scale" not in blocks.input_names:
                if "guider" in self._pipeline.component_names and "guider" not in component_updates:
                    guider_spec = self._pipeline.get_component_spec("guider")
                    guider = guider_spec.create(guidance_scale=value)
                    self._pipeline.update_components(guider=guider)

            # if a dict is passed and is not an pipeline input, we unpack and process its contents
            # e.g. `embeddings` from text_encoder node
            elif isinstance(value, dict) and name not in blocks.input_names:
                for k, v in value.items():
                    if k in blocks.input_names:
                        node_kwargs[k] = v
                    else:
                        expected_inputs = "\n  - ".join(blocks.input_names)
                        logger.warning(
                            f"Input '{name}:{k}' is not expected by {self.node_type} blocks.\n"
                            f"Expected inputs:\n  - {expected_inputs} \n"
                            f"Blocks: {blocks}"
                        )
            # pass the value as it is to the pipeline
            else:
                node_kwargs[name] = value

        if route_runtime_inputs is not None:
            node_kwargs["generator"] = route_runtime_inputs["generator"]
            if route_runtime_inputs["processed_mask_image"] is not None:
                if "processed_mask_image" not in blocks.input_names:
                    raise ValueError("The selected Modular denoiser does not expose the routed mask input.")
                node_kwargs["processed_mask_image"] = route_runtime_inputs["processed_mask_image"]
            for name in ("mask", "masked_image_latents", "crops_coords"):
                value = route_runtime_inputs.get(name)
                if value is None:
                    continue
                if name not in blocks.input_names:
                    raise ValueError(f"The selected Modular denoiser does not expose routed SDXL input '{name}'.")
                node_kwargs[name] = value
            if route_contract == "wan_i2v":
                for name in ("height", "width", "num_frames"):
                    if name not in blocks.input_names:
                        raise ValueError(f"The selected Wan denoiser does not expose routed input '{name}'.")
                    node_kwargs[name] = route_runtime_inputs[name]

        # Compatibility workaround: hidden height/width values may still be passed by older graphs.
        edit_models = [
            "Flux2KleinModularPipeline",
            "QwenImageEditModularPipeline",
            "QwenImageEditPlusModularPipeline",
            "FluxKontextModularPipeline",
        ]
        if (
            "image_latents" in node_kwargs
            and node_kwargs["image_latents"] is not None
            and self._model_type not in edit_models
        ):
            node_kwargs.pop("height", None)
            node_kwargs.pop("width", None)

        # 5. figure out the outputs to return based on node_config["output_names"]
        outputs = {}
        output_names = node_config["output_names"].copy()
        # "doc" is a standard node output but not a pipeline output
        if "doc" in output_names:
            output_names.remove("doc")
            outputs["doc"] = self._pipeline.blocks.doc
        route_output_declared = ROUTE_STATE_OUTPUT in output_names
        if route_output_declared:
            output_names.remove(ROUTE_STATE_OUTPUT)
            outputs[ROUTE_STATE_OUTPUT] = None
        pipeline_output_names = list(output_names)
        if route_uses_hidden_denoise_mask(self._model_type) and "mask" not in pipeline_output_names:
            pipeline_output_names.append("mask")

        if supported_route_model:
            self._require_route_component_inputs(
                kwargs,
                model_input_names=model_input_names,
                bundle_names=bundle_names,
            )
            installed_vae = self._require_installed_route_components(
                kwargs,
                component_updates,
                model_input_names=model_input_names,
            )
            if route_geometry is not None:
                live_vae, live_geometry = self._resolve_sdxl_route_vae(
                    kwargs,
                    require_resident_pipeline=True,
                )
                if live_vae is not installed_vae or live_geometry != route_geometry:
                    raise ValueError("The connected Denoise VAE changed before upstream execution.")
            elif route_contract == "wan_i2v":
                live_vae, live_transformer = self._resolve_wan_route_components(
                    kwargs,
                    require_resident_pipeline=True,
                )
                if live_vae is not installed_vae or live_vae is not preinit_vae:
                    raise ValueError("The connected Wan Denoise VAE changed before upstream execution.")
                if live_transformer is not route_transformer or live_transformer is not preinit_transformer:
                    raise ValueError("The connected Wan transformer changed before upstream execution.")
            if route_state is not None:
                validate_denoise_route_state(
                    route_state,
                    binding=route_binding,
                    model_type=self._model_type,
                    seed=kwargs["seed"],
                    image_latents=effective_image_latents,
                    mask=effective_mask,
                    masked_image_latents=effective_masked_image_latents,
                    vae_component=installed_vae,
                    vae_latent_channels=(route_geometry[0] if route_geometry is not None else None),
                    vae_scale_factor=(route_geometry[1] if route_geometry is not None else None),
                    control_image_latents=effective_control_latents,
                    controlnet_component=effective_controlnet_component,
                    controlnet_bundle_present=controlnet_bundle_present,
                    ip_adapter_present=kwargs.get("ip_adapter") is not None,
                    image_embeds=effective_image_embeds,
                    image_condition_latents=effective_image_condition_latents,
                    height=kwargs.get("height"),
                    width=kwargs.get("width"),
                    num_frames=kwargs.get("num_frames"),
                    transformer_component=route_transformer,
                )

        # 6. run the pipeline and update the outputs dict with the pipeline outputs
        transformer = getattr(self._pipeline, "transformer", None)
        signature_state = restore_wrapped_forward_signature(transformer) if transformer is not None else None
        self._active_pipeline = self._pipeline
        self._publish_initial_denoise_progress(num_inference_steps)
        try:
            node_outputs = self._pipeline(**node_kwargs, output=pipeline_output_names)
        except ValueError as e:
            self.notify(str(e), variant="error", persist=False, autoHideDuration=MESSAGE_DURATION)
            raise
        except KeyError as e:
            if embeddings_missing_error(e):
                self.notify(
                    MISSING_EMBEDDINGS_MESSAGE,
                    variant="error",
                    persist=False,
                    autoHideDuration=MESSAGE_DURATION,
                )
                raise RuntimeError(MISSING_EMBEDDINGS_MESSAGE) from e
            raise
        except AttributeError as e:
            # the config error should be the missing scheduler
            if "config" in str(e):
                self.notify(
                    "You have to connect the scheduler",
                    variant="error",
                    persist=False,
                    autoHideDuration=MESSAGE_DURATION,
                )
                raise RuntimeError("You have to connect the scheduler") from e

            # any other error just show the original message
            self.notify(str(e), variant="error", persist=False, autoHideDuration=MESSAGE_DURATION)
            raise
        finally:
            self._active_pipeline = None
            reset_wrapped_forward_signature(signature_state)

        for name in output_names:
            outputs[name] = node_outputs.get(name)
        if supported_route_model:
            self._require_route_component_inputs(
                kwargs,
                model_input_names=model_input_names,
                bundle_names=bundle_names,
            )
            installed_vae = self._require_installed_route_components(
                kwargs,
                component_updates,
                model_input_names=model_input_names,
            )
            if route_geometry is not None:
                live_vae, live_geometry = self._resolve_sdxl_route_vae(
                    kwargs,
                    require_resident_pipeline=True,
                )
                if live_vae is not installed_vae or live_geometry != route_geometry:
                    raise ValueError("The connected Denoise VAE changed during upstream execution.")
            elif route_contract == "wan_i2v":
                live_vae, live_transformer = self._resolve_wan_route_components(
                    kwargs,
                    require_resident_pipeline=True,
                )
                if live_vae is not installed_vae or live_vae is not preinit_vae:
                    raise ValueError("The connected Wan Denoise VAE changed during upstream execution.")
                if live_transformer is not route_transformer or live_transformer is not preinit_transformer:
                    raise ValueError("The connected Wan transformer changed during upstream execution.")
            if route_state is not None:
                validate_denoise_route_state(
                    route_state,
                    binding=route_binding,
                    model_type=self._model_type,
                    seed=kwargs["seed"],
                    image_latents=effective_image_latents,
                    mask=effective_mask,
                    masked_image_latents=effective_masked_image_latents,
                    vae_component=installed_vae,
                    vae_latent_channels=(route_geometry[0] if route_geometry is not None else None),
                    vae_scale_factor=(route_geometry[1] if route_geometry is not None else None),
                    control_image_latents=effective_control_latents,
                    controlnet_component=effective_controlnet_component,
                    controlnet_bundle_present=controlnet_bundle_present,
                    ip_adapter_present=kwargs.get("ip_adapter") is not None,
                    image_embeds=effective_image_embeds,
                    image_condition_latents=effective_image_condition_latents,
                    height=kwargs.get("height"),
                    width=kwargs.get("width"),
                    num_frames=kwargs.get("num_frames"),
                    transformer_component=route_transformer,
                )
            if not route_output_declared:
                raise ValueError("The backend Modular route contract is missing its Denoise route output.")
            if route_state is not None:
                outputs[ROUTE_STATE_OUTPUT] = issue_decode_route_state(
                    route_state,
                    binding=route_binding,
                    actual_mask=(
                        node_outputs.get("mask") if route_uses_hidden_denoise_mask(self._model_type) else None
                    ),
                    latents=outputs.get("latents"),
                    vae_component=(installed_vae if route_contract in {"sdxl", "wan_i2v"} else None),
                    transformer_component=(route_transformer if route_contract == "wan_i2v" else None),
                    execution_device=(device if route_contract == "wan_i2v" else None),
                )
            else:
                if route_uses_hidden_denoise_mask(self._model_type) and node_outputs.get("mask") is not None:
                    raise ValueError(
                        "A normal Modular Denoise route unexpectedly returned inpaint mask state."
                    )
                outputs[ROUTE_STATE_OUTPUT] = issue_normal_decode_route_state(
                    binding=route_binding,
                    latents=outputs.get("latents"),
                    vae_component=(installed_vae if route_geometry is not None else None),
                    vae_latent_channels=(route_geometry[0] if route_geometry is not None else None),
                    vae_scale_factor=(route_geometry[1] if route_geometry is not None else None),
                )

        return outputs
