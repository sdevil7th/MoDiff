# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import importlib
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
    DummyCustomPipeline,
    pipeline_class_from_runtime_inputs,
    pipeline_class_to_modiff_node_config,
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
            return None

        if model_type is None or model_type == "" or model_type == "DummyCustomPipeline":
            self._pipeline_class = DummyCustomPipeline
        else:
            diffusers_module = importlib.import_module("diffusers")
            self._pipeline_class = getattr(diffusers_module, model_type)

        self._model_type = model_type

        _, node_config = pipeline_class_to_modiff_node_config(self._pipeline_class, self.node_type)
        # not support this node type
        if node_config is None:
            self.send_node_definition(node_params)
            return

        node_params_to_update = node_config["params"]
        node_params_to_update.pop("unet", None)

        node_params.update(**node_params_to_update)
        self.send_node_definition(node_params)

    def __init__(self, node_id=None):
        super().__init__(node_id)
        self._model_type = ""
        self._pipeline_class = None

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
        self._pipeline_class = pipeline_class_from_runtime_inputs(self._pipeline_class, kwargs)

        if not ((unet := kwargs.get("unet")) and isinstance(unet, dict)):
            self.notify(
                "You have to connect the denoise model",
                variant="error",
                persist=False,
                autoHideDuration=MESSAGE_DURATION,
            )
            return None

        # 1. Get node config
        blocks, node_config = pipeline_class_to_modiff_node_config(self._pipeline_class, self.node_type)

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
        repo_id = unet.get("repo_id", None)

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
        self._pipeline = runtime_blocks.init_pipeline(repo_id, components_manager=components)

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
        model_ids = collect_model_ids(
            kwargs,
            target_key_names=model_input_names,
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

        device = self._pipeline._execution_device

        # 4. compile a dict of runtime inputs from kwargs based on node_config["input_names"]
        node_kwargs = {}
        input_names = node_config["input_names"]

        for name in input_names:
            value = kwargs.get(name)
            if value is None:
                continue

            # special case #1: `seed` -> always create a `generator`
            if name == "seed":
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

        # 6. run the pipeline and update the outputs dict with the pipeline outputs
        transformer = getattr(self._pipeline, "transformer", None)
        signature_state = restore_wrapped_forward_signature(transformer) if transformer is not None else None
        self._active_pipeline = self._pipeline
        self._publish_initial_denoise_progress(num_inference_steps)
        try:
            node_outputs = self._pipeline(**node_kwargs, output=output_names)
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

        outputs.update(node_outputs)

        return outputs
