"""Official whole-workflow Modular Diffusers block executors.

These nodes run package-owned ``ModularPipelineBlocks`` through
``init_pipeline()`` and the normal ``ModularPipeline.__call__`` interface. They
do not invoke a block's internal ``(components, state)`` protocol and do not
reimplement upstream scheduler or loop logic.
"""

from __future__ import annotations

import json
import math
import re
import threading
import weakref

from PIL import Image as PILImage

from modiff.ltx25_execution_contract import (
    configure_ltx25_diffusion_decoder,
    configure_ltx25_distilled_denoise_components,
    ltx25_distilled_denoise_kwargs,
)
from modiff.modular_whole_workflow_contracts import reviewed_whole_workflow_graph_adapter
from modiff.NodeBase import NodeBase

from . import components
from .modular_utils import modular_generator_from_seed, pipeline_class_from_model_type
from .route_state import require_component_binding
from .utils import collect_model_ids
from .workflow_runtime import continuation_generator_from_seed


_ISSUER = object()
_STATE_LOCK = threading.RLock()
_ISSUED_WORKFLOW_STATES = weakref.WeakSet()

_MINIMAX_PIPELINE = "MiniMaxMusic3ModularPipeline"
_MINIMAX_WORKFLOW = "default"
_ANIMA_PIPELINE = "AnimaModularPipeline"
_HELIOS_BASE_PIPELINE = "HeliosModularPipeline"
_HELIOS_PYRAMID_PIPELINE = "HeliosPyramidModularPipeline"
_HELIOS_DISTILLED_PIPELINE = "HeliosPyramidDistilledModularPipeline"
_HELIOS_PIPELINES = frozenset({_HELIOS_BASE_PIPELINE, _HELIOS_PYRAMID_PIPELINE, _HELIOS_DISTILLED_PIPELINE})
_WAN_ANIMATE_BASE_PIPELINE = "WanAnimate2ModularPipeline"
_WAN_ANIMATE_DISTILLED_PIPELINE = "WanAnimate2DistilledModularPipeline"
_WAN_ANIMATE_PIPELINES = frozenset({_WAN_ANIMATE_BASE_PIPELINE, _WAN_ANIMATE_DISTILLED_PIPELINE})
_WAN_ANIMATE_WORKFLOW = "default"
_HUNYUAN_VIDEO_15_PIPELINE = "HunyuanVideo15ModularPipeline"
_HUNYUAN_VIDEO_15_WORKFLOWS = frozenset({"text2video", "image2video"})
_STABLE_DIFFUSION_3_PIPELINE = "StableDiffusion3ModularPipeline"
_STABLE_DIFFUSION_3_WORKFLOWS = frozenset({"text2image", "image2image"})
_KREA_2_PIPELINE = "Krea2ModularPipeline"
_KREA_2_TURBO_PIPELINE = "Krea2TurboModularPipeline"
_KREA_2_PIPELINES = frozenset({_KREA_2_PIPELINE, _KREA_2_TURBO_PIPELINE})
_KREA_2_WORKFLOW = "text2image"
_IDEOGRAM_4_PIPELINE = "Ideogram4ModularPipeline"
_IDEOGRAM_4_WORKFLOW = "text2image"
_COSMOS3_DISTILLED_PIPELINE = "Cosmos3DistilledModularPipeline"
_COSMOS3_DISTILLED_WORKFLOWS = frozenset({"text2image", "text2video", "image2video", "video2video"})
_COSMOS3_OMNI_PIPELINE = "Cosmos3OmniModularPipeline"
_COSMOS3_OMNI_WORKFLOWS = frozenset(
    {
        "text2image",
        "text2video",
        "image2video",
        "video2video",
        "text2video_with_sound",
        "image2video_with_sound",
        "video2video_with_sound",
        "action_policy",
        "action_forward_dynamics",
        "action_inverse_dynamics",
    }
)
_COSMOS3_OMNI_SOUND_WORKFLOWS = frozenset(
    {"text2video_with_sound", "image2video_with_sound", "video2video_with_sound"}
)
_COSMOS3_OMNI_ACTION_MODES = {
    "action_policy": "policy",
    "action_forward_dynamics": "forward_dynamics",
    "action_inverse_dynamics": "inverse_dynamics",
}
_MINIMAX_H3_PIPELINE = "MiniMaxH3ModularPipeline"
_MINIMAX_H3_WORKFLOWS = frozenset({"t2va", "fl2va", "ref2va"})
_LTX_25_PIPELINE = "LTX25ModularPipeline"
_LTX_25_WORKFLOWS = frozenset({"text2video", "image2video", "condition", "in_context"})


class _WorkflowState:
    """Sealed, process-local continuation state between official block nodes."""

    __slots__ = (
        "_pipeline_token",
        "_pipeline_class",
        "_workflow_id",
        "_completed_stage",
        "_state",
        "_sealed",
        "__weakref__",
    )

    def __init__(
        self,
        issuer,
        *,
        pipeline_token,
        pipeline_class,
        workflow_id,
        completed_stage,
        state,
    ):
        if issuer is not _ISSUER:
            raise TypeError("Modular workflow states are issued only by official block nodes.")
        object.__setattr__(self, "_pipeline_token", pipeline_token)
        object.__setattr__(self, "_pipeline_class", pipeline_class)
        object.__setattr__(self, "_workflow_id", workflow_id)
        object.__setattr__(self, "_completed_stage", completed_stage)
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, _name, _value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Modular workflow states are immutable.")
        object.__setattr__(self, _name, _value)

    def __copy__(self):
        return self

    def __deepcopy__(self, _memo):
        return self

    def __reduce_ex__(self, _protocol):
        raise TypeError("Modular workflow states cannot be serialized.")


def _issue_workflow_state(*, token, pipeline_class, workflow_id, completed_stage, state):
    issued = _WorkflowState(
        _ISSUER,
        pipeline_token=token,
        pipeline_class=pipeline_class,
        workflow_id=workflow_id,
        completed_stage=completed_stage,
        state=state,
    )
    with _STATE_LOCK:
        _ISSUED_WORKFLOW_STATES.add(issued)
    return issued


def _require_workflow_state(value, *, token, pipeline_class, workflow_id, completed_stage):
    with _STATE_LOCK:
        issued = type(value) is _WorkflowState and value in _ISSUED_WORKFLOW_STATES
    if not issued:
        raise ValueError(
            "The connected Modular workflow state is not a current process-local state. "
            "Rerun the preceding official block node."
        )
    if value._pipeline_token is not token:
        raise ValueError("The connected Modular workflow state belongs to a different Models Loader execution.")
    if value._pipeline_class != pipeline_class or value._workflow_id != workflow_id:
        raise ValueError("The connected Modular workflow state belongs to a different pipeline workflow.")
    if value._completed_stage != completed_stage:
        raise ValueError(
            f"The connected Modular workflow state completed {value._completed_stage!r}, "
            f"not required stage {completed_stage!r}."
        )
    return value._state


def _require_exact_string(value, *, label, expected):
    if type(value) is not str or value != expected:
        raise ValueError(f"{label} must be the exact reviewed value {expected!r}.")
    return value


def _reviewed_action_stage(pipeline_class, workflow_id, action_key):
    if type(pipeline_class) is not str or not pipeline_class:
        raise ValueError("Pipeline class must be an exact reviewed string.")
    if type(workflow_id) is not str or not workflow_id:
        raise ValueError("Workflow must be an exact reviewed string.")
    contract = reviewed_whole_workflow_graph_adapter(pipeline_class, workflow_id)
    if contract is None:
        raise ValueError(f"{pipeline_class}/{workflow_id} has no reviewed official whole-workflow execution adapter.")
    try:
        action_index = contract["actionSequence"].index(action_key)
    except ValueError as error:
        raise ValueError(
            f"The reviewed {pipeline_class}/{workflow_id} adapter has no action {action_key!r}."
        ) from error
    return contract, action_index, contract["upstreamBlockSequence"][action_index]


def _reviewed_preceding_stage(pipeline_class, workflow_id, action_key):
    contract, action_index, _stage = _reviewed_action_stage(pipeline_class, workflow_id, action_key)
    if action_index == 0:
        return None
    return contract["upstreamBlockSequence"][action_index - 1]


class _OfficialWorkflowBlockMixin:
    stage = ""
    action_key = ""

    def _prepare_pipeline(self, *, pipeline_components, pipeline_class, workflow_id, block_path):
        _contract, _action_index, expected_stage = _reviewed_action_stage(
            pipeline_class,
            workflow_id,
            self.action_key,
        )
        _require_exact_string(block_path, label="Block path", expected=expected_stage)
        if self.stage != expected_stage:
            raise RuntimeError("The official Modular workflow node does not match its reviewed stage contract.")

        token = require_component_binding(
            pipeline_components,
            label="pipeline component bundle",
            expected_model_type=pipeline_class,
            expected_role="pipeline_components",
        )
        pipeline_type = pipeline_class_from_model_type(pipeline_class)
        definition = pipeline_type()
        blocks = definition.blocks
        if workflow_id != "default":
            blocks = blocks.get_workflow(workflow_id)
        block = blocks.sub_blocks.get(block_path)
        if block is None:
            raise ValueError(
                f"The reviewed Modular workflow {pipeline_class}/{workflow_id} has no block {block_path!r}."
            )
        pipeline = block.init_pipeline(components_manager=components)
        expected_components = tuple(pipeline.pretrained_component_names)
        model_ids = collect_model_ids(
            {"pipeline_components": pipeline_components},
            target_key_names=("pipeline_components",),
            target_model_names=expected_components,
        )
        installed = components.get_components_by_ids(ids=model_ids, return_dict_with_names=True) if model_ids else {}
        missing = sorted(set(expected_components) - set(installed))
        if missing:
            raise ValueError(
                "The connected Models Loader bundle is missing required official block components: "
                + ", ".join(missing)
            )
        if installed:
            pipeline.update_components(**installed)
        return token, pipeline


class WorkflowSemanticGeneration(_OfficialWorkflowBlockMixin, NodeBase):
    """Run MiniMax Music 3's official semantic-generation block."""

    label = "Modular Semantic Generation"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "semantic_generator"
    action_key = "semantic_generator"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "MiniMaxMusic3ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "default", "hidden": True},
        "block_path": {"type": "string", "default": "semantic_generator", "hidden": True},
        "prompt": {"label": "Music Description", "display": "textarea", "type": "text", "default": ""},
        "lyrics": {"label": "Lyrics", "display": "textarea", "type": "text", "default": ""},
        "audio_duration": {
            "label": "Duration",
            "type": "float",
            "default": 60.0,
            "min": 1.0,
            "max": 360.0,
        },
        "seed": {
            "label": "Seed",
            "display": "random",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 4294967295,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        token, pipeline = self._prepare_pipeline(
            pipeline_components=kwargs.get("pipeline_components"),
            pipeline_class=kwargs.get("pipeline_class"),
            workflow_id=kwargs.get("workflow_id"),
            block_path=kwargs.get("block_path"),
        )
        prompt = kwargs.get("prompt")
        lyrics = kwargs.get("lyrics")
        if type(prompt) is not str or not prompt.strip():
            raise ValueError("MiniMax Music 3 requires a nonblank music description.")
        if type(lyrics) is not str or not lyrics.strip():
            raise ValueError("MiniMax Music 3 requires nonblank lyrics.")
        duration = float(kwargs.get("audio_duration", 60.0))
        if not 1.0 <= duration <= 360.0:
            raise ValueError("MiniMax Music 3 duration must be from 1 through 360 seconds.")
        generator = modular_generator_from_seed(kwargs.get("seed", 0), pipeline)
        state = pipeline(
            prompt=prompt,
            lyrics=lyrics,
            audio_duration=duration,
            generator=generator,
        )
        return {
            "state_out": _issue_workflow_state(
                token=token,
                pipeline_class=_MINIMAX_PIPELINE,
                workflow_id=_MINIMAX_WORKFLOW,
                completed_stage=self.stage,
                state=state,
            )
        }


class WorkflowDenoise(_OfficialWorkflowBlockMixin, NodeBase):
    """Run the official denoise block using a sealed preceding state."""

    label = "Modular Workflow Denoise"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "denoise"
    action_key = "workflow_denoise"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "MiniMaxMusic3ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "default", "hidden": True},
        "block_path": {"type": "string", "default": "denoise", "hidden": True},
        "num_inference_steps": {
            "label": "Steps",
            "display": "slider",
            "type": "int",
            "default": 30,
            "min": 1,
            "max": 100,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        token, pipeline = self._prepare_pipeline(
            pipeline_components=kwargs.get("pipeline_components"),
            pipeline_class=kwargs.get("pipeline_class"),
            workflow_id=kwargs.get("workflow_id"),
            block_path=kwargs.get("block_path"),
        )
        steps = _bounded_integer(
            kwargs.get("num_inference_steps", 30),
            label="MiniMax Music 3 denoise steps",
            minimum=1,
            maximum=100,
        )
        state = _require_workflow_state(
            kwargs.get("state_in"),
            token=token,
            pipeline_class=_MINIMAX_PIPELINE,
            workflow_id=_MINIMAX_WORKFLOW,
            completed_stage="semantic_generator",
        )
        state = pipeline(state=state, num_inference_steps=steps)
        return {
            "state_out": _issue_workflow_state(
                token=token,
                pipeline_class=_MINIMAX_PIPELINE,
                workflow_id=_MINIMAX_WORKFLOW,
                completed_stage=self.stage,
                state=state,
            )
        }


class WorkflowDecodeAudio(_OfficialWorkflowBlockMixin, NodeBase):
    """Run the official decoder and convert its waveform to MoDiff audio."""

    label = "Modular Workflow Decode Audio"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "decode"
    action_key = "workflow_audio_decoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "MiniMaxMusic3ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "default", "hidden": True},
        "block_path": {"type": "string", "default": "decode", "hidden": True},
        "audio": {"label": "Audio", "display": "output", "type": "audio"},
        "sample_rate": {"label": "Sample Rate", "display": "output", "type": "int"},
        "duration_seconds": {"label": "Duration", "display": "output", "type": "float"},
    }

    def execute(self, **kwargs):
        token, pipeline = self._prepare_pipeline(
            pipeline_components=kwargs.get("pipeline_components"),
            pipeline_class=kwargs.get("pipeline_class"),
            workflow_id=kwargs.get("workflow_id"),
            block_path=kwargs.get("block_path"),
        )
        state = _require_workflow_state(
            kwargs.get("state_in"),
            token=token,
            pipeline_class=_MINIMAX_PIPELINE,
            workflow_id=_MINIMAX_WORKFLOW,
            completed_stage="denoise",
        )
        state = pipeline(state=state, output_type="pt")
        raw_audio = state.get("audios")
        from modules.DiffusersAudio.main import output_to_audio_object

        sample_rate = int(getattr(pipeline, "sampling_rate", None) or 44_100)
        audio = output_to_audio_object(raw_audio, sample_rate=sample_rate)
        return {
            "audio": audio,
            "sample_rate": sample_rate,
            "duration_seconds": float(audio.get("duration_seconds") or 0.0),
        }


class WorkflowTextEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run an official package-owned text-encoder block and seal its state."""

    label = "Modular Workflow Text Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "text_encoder"
    action_key = "workflow_text_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "AnimaModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2image", "hidden": True},
        "block_path": {"type": "string", "default": "text_encoder", "hidden": True},
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "negative_prompt": {
            "label": "Negative Prompt",
            "display": "textarea",
            "type": "text",
            "default": "",
        },
        "max_sequence_length": {
            "label": "Maximum Sequence Length",
            "type": "int",
            "default": 512,
            "min": 1,
            "max": 4096,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        pipeline_class = kwargs.get("pipeline_class")
        workflow_id = kwargs.get("workflow_id")
        token, pipeline = self._prepare_pipeline(
            pipeline_components=kwargs.get("pipeline_components"),
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            block_path=kwargs.get("block_path"),
        )
        prompt = kwargs.get("prompt")
        negative_prompt = kwargs.get("negative_prompt", "")
        if type(prompt) is not str or not prompt.strip():
            raise ValueError("The official Modular workflow requires a nonblank prompt.")
        if type(negative_prompt) is not str:
            raise ValueError("The official Modular workflow negative prompt must be text.")
        max_sequence_length = kwargs.get("max_sequence_length", 512)
        max_sequence_limit = 512 if pipeline_class in _HELIOS_PIPELINES else 4096
        if (
            isinstance(max_sequence_length, bool)
            or not isinstance(max_sequence_length, int)
            or not 1 <= max_sequence_length <= max_sequence_limit
        ):
            raise ValueError(
                f"The official Modular workflow maximum sequence length must be an integer from 1 through "
                f"{max_sequence_limit}."
            )
        state = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_sequence_length=max_sequence_length,
        )
        return {
            "state_out": _issue_workflow_state(
                token=token,
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                completed_stage=self.stage,
                state=state,
            )
        }


class WorkflowImageEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run an official package-owned image VAE-encoder block."""

    label = "Modular Workflow Image Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "vae_encoder"
    action_key = "workflow_image_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "AnimaModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "img2img", "hidden": True},
        "block_path": {"type": "string", "default": "vae_encoder", "hidden": True},
        "image": {"label": "Image", "display": "input", "type": "image", "required": True},
        "width": {"label": "Width", "type": "int", "default": 1024, "min": 512, "max": 1536, "step": 8},
        "height": {
            "label": "Height",
            "type": "int",
            "default": 1024,
            "min": 512,
            "max": 1536,
            "step": 8,
        },
        "seed": {
            "label": "Seed",
            "display": "random",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 4294967295,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        pipeline_class = kwargs.get("pipeline_class")
        workflow_id = kwargs.get("workflow_id")
        token, pipeline = self._prepare_pipeline(
            pipeline_components=kwargs.get("pipeline_components"),
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            block_path=kwargs.get("block_path"),
        )
        image = kwargs.get("image")
        if not isinstance(image, PILImage.Image):
            raise ValueError("Anima image-to-image requires one exact PIL image.")
        width = kwargs.get("width", 1024)
        height = kwargs.get("height", 1024)
        for label, value in (("width", width), ("height", height)):
            if isinstance(value, bool) or not isinstance(value, int) or not 512 <= value <= 1536 or value % 8:
                raise ValueError(f"Anima {label} must be a multiple of 8 from 512 through 1536.")
        previous_stage = _reviewed_preceding_stage(pipeline_class, workflow_id, self.action_key)
        state = _require_workflow_state(
            kwargs.get("state_in"),
            token=token,
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            completed_stage=previous_stage,
        )
        generator = modular_generator_from_seed(kwargs.get("seed", 0), pipeline)
        state = pipeline(
            state=state,
            image=image,
            width=width,
            height=height,
            generator=generator,
        )
        return {
            "state_out": _issue_workflow_state(
                token=token,
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                completed_stage=self.stage,
                state=state,
            )
        }


class WorkflowImageDenoise(_OfficialWorkflowBlockMixin, NodeBase):
    """Run an official package-owned image denoise block over sealed state."""

    label = "Modular Workflow Image Denoise"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "denoise"
    action_key = "workflow_image_denoise"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "AnimaModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2image", "hidden": True},
        "block_path": {"type": "string", "default": "denoise", "hidden": True},
        "width": {"label": "Width", "type": "int", "default": 1024, "min": 512, "max": 1536, "step": 8},
        "height": {
            "label": "Height",
            "type": "int",
            "default": 1024,
            "min": 512,
            "max": 1536,
            "step": 8,
        },
        "num_images_per_prompt": {
            "label": "Images per Prompt",
            "type": "int",
            "default": 1,
            "min": 1,
            "max": 4,
        },
        "num_inference_steps": {
            "label": "Steps",
            "display": "slider",
            "type": "int",
            "default": 50,
            "min": 1,
            "max": 100,
        },
        "guidance_scale": {
            "label": "Guidance Scale",
            "display": "slider",
            "type": "float",
            "default": 4.0,
            "min": 1.0,
            "max": 20.0,
            "step": 0.1,
        },
        "strength": {
            "label": "Strength",
            "display": "slider",
            "type": "float",
            "default": 0.9,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
        },
        "seed": {
            "label": "Seed",
            "display": "random",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 4294967295,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        pipeline_class = kwargs.get("pipeline_class")
        workflow_id = kwargs.get("workflow_id")
        token, pipeline = self._prepare_pipeline(
            pipeline_components=kwargs.get("pipeline_components"),
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            block_path=kwargs.get("block_path"),
        )
        previous_stage = _reviewed_preceding_stage(pipeline_class, workflow_id, self.action_key)
        state = _require_workflow_state(
            kwargs.get("state_in"),
            token=token,
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            completed_stage=previous_stage,
        )
        width = kwargs.get("width", 1024)
        height = kwargs.get("height", 1024)
        for label, value in (("width", width), ("height", height)):
            if isinstance(value, bool) or not isinstance(value, int) or not 512 <= value <= 1536 or value % 8:
                raise ValueError(f"Anima {label} must be a multiple of 8 from 512 through 1536.")
        steps = kwargs.get("num_inference_steps", 50)
        images_per_prompt = kwargs.get("num_images_per_prompt", 1)
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 100:
            raise ValueError("Anima denoise steps must be an integer from 1 through 100.")
        if (
            isinstance(images_per_prompt, bool)
            or not isinstance(images_per_prompt, int)
            or not 1 <= images_per_prompt <= 4
        ):
            raise ValueError("Anima images per prompt must be an integer from 1 through 4.")
        guidance_scale = float(kwargs.get("guidance_scale", 4.0))
        if not 1.0 <= guidance_scale <= 20.0:
            raise ValueError("Anima guidance scale must be from 1 through 20.")
        if getattr(pipeline, "guider", None) is None or not callable(getattr(pipeline.guider, "new", None)):
            raise ValueError("The official Anima denoise block is missing its ClassifierFreeGuidance component.")
        pipeline.update_components(guider=pipeline.guider.new(guidance_scale=guidance_scale))
        generator = modular_generator_from_seed(kwargs.get("seed", 0), pipeline)
        call_kwargs = {
            "state": state,
            "width": width,
            "height": height,
            "num_images_per_prompt": images_per_prompt,
            "num_inference_steps": steps,
            "generator": generator,
        }
        if workflow_id == "img2img":
            strength = float(kwargs.get("strength", 0.9))
            if not 0.0 < strength <= 1.0:
                raise ValueError("Anima image-to-image strength must be greater than 0 and at most 1.")
            call_kwargs["strength"] = strength
        state = pipeline(**call_kwargs)
        return {
            "state_out": _issue_workflow_state(
                token=token,
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                completed_stage=self.stage,
                state=state,
            )
        }


class WorkflowDecodeImage(_OfficialWorkflowBlockMixin, NodeBase):
    """Run an official package-owned image decoder over sealed denoised state."""

    label = "Modular Workflow Decode Image"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "decode"
    action_key = "workflow_image_decoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "AnimaModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2image", "hidden": True},
        "block_path": {"type": "string", "default": "decode", "hidden": True},
        "images": {"label": "Images", "display": "output", "type": "image"},
    }

    def execute(self, **kwargs):
        pipeline_class = kwargs.get("pipeline_class")
        workflow_id = kwargs.get("workflow_id")
        token, pipeline = self._prepare_pipeline(
            pipeline_components=kwargs.get("pipeline_components"),
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            block_path=kwargs.get("block_path"),
        )
        previous_stage = _reviewed_preceding_stage(pipeline_class, workflow_id, self.action_key)
        state = _require_workflow_state(
            kwargs.get("state_in"),
            token=token,
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            completed_stage=previous_stage,
        )
        state = pipeline(state=state, output_type="pil")
        images = state.get("images")
        if (
            not isinstance(images, list)
            or not images
            or not all(isinstance(image, PILImage.Image) for image in images)
        ):
            raise ValueError("The official Anima decoder did not return PIL images.")
        return {"images": images}


def _require_helios_pipeline(pipeline_class):
    if pipeline_class not in _HELIOS_PIPELINES:
        raise ValueError("The official video workflow node requires an exact reviewed Helios pipeline class.")


def _helios_dimensions(width, height):
    for label, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or not 256 <= value <= 1024 or value % 8:
            raise ValueError(f"Helios {label} must be a multiple of 8 from 256 through 1024.")


def _bounded_integer(value, *, label, minimum, maximum):
    # NumberField commits canonical text because it must preserve an in-flight
    # editable value in the browser. Parse that transport form at the node
    # boundary while retaining strict integer/range validation. Do not accept
    # booleans, fractions, whitespace variants, or non-canonical leading
    # zeroes.
    if isinstance(value, str) and re.fullmatch(r"(?:0|-?[1-9][0-9]*)", value):
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer from {minimum} through {maximum}.")
    return value


class WorkflowVideoImageEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run Helios' official image VAE-encoder block."""

    label = "Modular Workflow Video Image Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "vae_encoder"
    action_key = "workflow_video_image_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "HeliosModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "image2video", "hidden": True},
        "block_path": {"type": "string", "default": "vae_encoder", "hidden": True},
        "image": {"label": "Image", "display": "input", "type": "image", "required": True},
        "width": {"label": "Width", "type": "int", "default": 640, "min": 256, "max": 1024, "step": 8},
        "height": {"label": "Height", "type": "int", "default": 384, "min": 256, "max": 1024, "step": 8},
        "num_latent_frames_per_chunk": {
            "label": "Latent Frames per Chunk",
            "type": "int",
            "default": 9,
            "min": 2,
            "max": 17,
        },
        "seed": {
            "label": "Seed",
            "display": "random",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 4294967295,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        pipeline_class = kwargs.get("pipeline_class")
        workflow_id = kwargs.get("workflow_id")
        _require_helios_pipeline(pipeline_class)
        token, pipeline = self._prepare_pipeline(
            pipeline_components=kwargs.get("pipeline_components"),
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            block_path=kwargs.get("block_path"),
        )
        image = kwargs.get("image")
        if not isinstance(image, PILImage.Image):
            raise ValueError("Helios image-to-video requires one exact PIL image.")
        width = kwargs.get("width", 640)
        height = kwargs.get("height", 384)
        _helios_dimensions(width, height)
        latent_frames = _bounded_integer(
            kwargs.get("num_latent_frames_per_chunk", 9),
            label="Helios latent frames per chunk",
            minimum=2,
            maximum=17,
        )
        previous_stage = _reviewed_preceding_stage(pipeline_class, workflow_id, self.action_key)
        state = _require_workflow_state(
            kwargs.get("state_in"),
            token=token,
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            completed_stage=previous_stage,
        )
        state = pipeline(
            state=state,
            image=image,
            width=width,
            height=height,
            num_latent_frames_per_chunk=latent_frames,
            generator=modular_generator_from_seed(kwargs.get("seed", 0), pipeline),
        )
        return {
            "state_out": _issue_workflow_state(
                token=token,
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                completed_stage=self.stage,
                state=state,
            )
        }


class WorkflowVideoEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run Helios' official video VAE-encoder block."""

    label = "Modular Workflow Video Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "vae_encoder"
    action_key = "workflow_video_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "HeliosModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "video2video", "hidden": True},
        "block_path": {"type": "string", "default": "vae_encoder", "hidden": True},
        "video": {"label": "Video", "display": "input", "type": "video", "required": True},
        "width": {"label": "Width", "type": "int", "default": 640, "min": 256, "max": 1024, "step": 8},
        "height": {"label": "Height", "type": "int", "default": 384, "min": 256, "max": 1024, "step": 8},
        "num_latent_frames_per_chunk": {
            "label": "Latent Frames per Chunk",
            "type": "int",
            "default": 9,
            "min": 2,
            "max": 17,
        },
        "seed": {
            "label": "Seed",
            "display": "random",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 4294967295,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        pipeline_class = kwargs.get("pipeline_class")
        workflow_id = kwargs.get("workflow_id")
        _require_helios_pipeline(pipeline_class)
        token, pipeline = self._prepare_pipeline(
            pipeline_components=kwargs.get("pipeline_components"),
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            block_path=kwargs.get("block_path"),
        )
        video = kwargs.get("video")
        if (
            not isinstance(video, list)
            or len(video) < 5
            or len(video) > 528
            or not all(isinstance(frame, PILImage.Image) for frame in video)
        ):
            raise ValueError("Helios video-to-video requires 5 through 528 decoded PIL frames.")
        width = kwargs.get("width", 640)
        height = kwargs.get("height", 384)
        _helios_dimensions(width, height)
        latent_frames = _bounded_integer(
            kwargs.get("num_latent_frames_per_chunk", 9),
            label="Helios latent frames per chunk",
            minimum=2,
            maximum=17,
        )
        minimum_frames = (latent_frames - 1) * 4 + 1
        if len(video) < minimum_frames:
            raise ValueError(
                f"Helios video-to-video needs at least {minimum_frames} frames for the selected latent chunk size."
            )
        previous_stage = _reviewed_preceding_stage(pipeline_class, workflow_id, self.action_key)
        state = _require_workflow_state(
            kwargs.get("state_in"),
            token=token,
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            completed_stage=previous_stage,
        )
        state = pipeline(
            state=state,
            video=video,
            width=width,
            height=height,
            num_latent_frames_per_chunk=latent_frames,
            generator=modular_generator_from_seed(kwargs.get("seed", 0), pipeline),
        )
        return {
            "state_out": _issue_workflow_state(
                token=token,
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                completed_stage=self.stage,
                state=state,
            )
        }


class WorkflowVideoDenoise(_OfficialWorkflowBlockMixin, NodeBase):
    """Run a reviewed Helios base, pyramid, or distilled denoise block."""

    label = "Modular Workflow Video Denoise"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "denoise"
    action_key = "workflow_video_denoise"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "HeliosModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2video", "hidden": True},
        "block_path": {"type": "string", "default": "denoise", "hidden": True},
        "width": {"label": "Width", "type": "int", "default": 640, "min": 256, "max": 1024, "step": 8},
        "height": {"label": "Height", "type": "int", "default": 384, "min": 256, "max": 1024, "step": 8},
        "num_frames": {"label": "Frames", "type": "int", "default": 132, "min": 33, "max": 264, "step": 33},
        "num_latent_frames_per_chunk": {
            "label": "Latent Frames per Chunk",
            "type": "int",
            "default": 9,
            "min": 2,
            "max": 17,
        },
        "history_size_long": {"label": "Long History", "type": "int", "default": 16, "min": 1, "max": 64},
        "history_size_mid": {"label": "Mid History", "type": "int", "default": 2, "min": 1, "max": 64},
        "history_size_short": {"label": "Short History", "type": "int", "default": 1, "min": 1, "max": 64},
        "keep_first_frame": {"label": "Keep First Frame", "type": "boolean", "default": True},
        "num_inference_steps": {"label": "Steps", "type": "int", "default": 50, "min": 1, "max": 100},
        "pyramid_stage_1_steps": {"label": "Pyramid Stage 1 Steps", "type": "int", "default": 10, "min": 1, "max": 50},
        "pyramid_stage_2_steps": {"label": "Pyramid Stage 2 Steps", "type": "int", "default": 10, "min": 1, "max": 50},
        "pyramid_stage_3_steps": {"label": "Pyramid Stage 3 Steps", "type": "int", "default": 10, "min": 1, "max": 50},
        "guidance_scale": {
            "label": "Guidance Scale",
            "display": "slider",
            "type": "float",
            "default": 5.0,
            "min": 1.0,
            "max": 20.0,
            "step": 0.1,
        },
        "image_noise_sigma_min": {
            "label": "Image Noise Sigma Min",
            "type": "float",
            "default": 0.111,
            "min": 0.0,
            "max": 1.0,
        },
        "image_noise_sigma_max": {
            "label": "Image Noise Sigma Max",
            "type": "float",
            "default": 0.135,
            "min": 0.0,
            "max": 1.0,
        },
        "video_noise_sigma_min": {
            "label": "Video Noise Sigma Min",
            "type": "float",
            "default": 0.111,
            "min": 0.0,
            "max": 1.0,
        },
        "video_noise_sigma_max": {
            "label": "Video Noise Sigma Max",
            "type": "float",
            "default": 0.135,
            "min": 0.0,
            "max": 1.0,
        },
        "is_amplify_first_chunk": {"label": "Amplify First Chunk", "type": "boolean", "default": True},
        "seed": {
            "label": "Seed",
            "display": "random",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 4294967295,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        pipeline_class = kwargs.get("pipeline_class")
        workflow_id = kwargs.get("workflow_id")
        _require_helios_pipeline(pipeline_class)
        token, pipeline = self._prepare_pipeline(
            pipeline_components=kwargs.get("pipeline_components"),
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            block_path=kwargs.get("block_path"),
        )
        previous_stage = _reviewed_preceding_stage(pipeline_class, workflow_id, self.action_key)
        state = _require_workflow_state(
            kwargs.get("state_in"),
            token=token,
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            completed_stage=previous_stage,
        )
        width = kwargs.get("width", 640)
        height = kwargs.get("height", 384)
        _helios_dimensions(width, height)
        num_frames = _bounded_integer(
            kwargs.get("num_frames", 132), label="Helios frame count", minimum=33, maximum=264
        )
        latent_frames = _bounded_integer(
            kwargs.get("num_latent_frames_per_chunk", 9),
            label="Helios latent frames per chunk",
            minimum=2,
            maximum=17,
        )
        history_sizes = [
            _bounded_integer(kwargs.get(name, default), label=f"Helios {label} history", minimum=1, maximum=64)
            for name, label, default in (
                ("history_size_long", "long", 16),
                ("history_size_mid", "mid", 2),
                ("history_size_short", "short", 1),
            )
        ]
        keep_first_frame = kwargs.get("keep_first_frame", True)
        if type(keep_first_frame) is not bool:
            raise ValueError("Helios keep-first-frame must be a boolean.")
        guidance_scale = float(kwargs.get("guidance_scale", 5.0))
        if not math.isfinite(guidance_scale) or not 1.0 <= guidance_scale <= 20.0:
            raise ValueError("Helios guidance scale must be finite and from 1 through 20.")
        if pipeline_class == _HELIOS_DISTILLED_PIPELINE and guidance_scale != 1.0:
            raise ValueError("Helios Distilled uses the official fixed guidance scale of 1.")
        if getattr(pipeline, "guider", None) is None or not callable(getattr(pipeline.guider, "new", None)):
            raise ValueError("The official Helios denoise block is missing its reviewed guider component.")
        pipeline.update_components(guider=pipeline.guider.new(guidance_scale=guidance_scale))
        call_kwargs = {
            "state": state,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "num_latent_frames_per_chunk": latent_frames,
            "history_sizes": history_sizes,
            "keep_first_frame": keep_first_frame,
            "generator": modular_generator_from_seed(kwargs.get("seed", 0), pipeline),
        }
        if pipeline_class == _HELIOS_BASE_PIPELINE:
            call_kwargs["num_inference_steps"] = _bounded_integer(
                kwargs.get("num_inference_steps", 50),
                label="Helios inference steps",
                minimum=1,
                maximum=100,
            )
        else:
            call_kwargs["pyramid_num_inference_steps_list"] = [
                _bounded_integer(
                    kwargs.get(f"pyramid_stage_{stage}_steps", 10),
                    label=f"Helios pyramid stage {stage} steps",
                    minimum=1,
                    maximum=50,
                )
                for stage in (1, 2, 3)
            ]
            if pipeline_class == _HELIOS_DISTILLED_PIPELINE:
                amplify = kwargs.get("is_amplify_first_chunk", True)
                if type(amplify) is not bool:
                    raise ValueError("Helios Distilled amplify-first-chunk must be a boolean.")
                call_kwargs["is_amplify_first_chunk"] = amplify
        if workflow_id in {"image2video", "video2video"}:
            sigma_values = {}
            for name in (
                "image_noise_sigma_min",
                "image_noise_sigma_max",
                "video_noise_sigma_min",
                "video_noise_sigma_max",
            ):
                value = float(kwargs.get(name, 0.111 if name.endswith("min") else 0.135))
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"Helios {name} must be finite and from 0 through 1.")
                sigma_values[name] = value
            if sigma_values["image_noise_sigma_min"] > sigma_values["image_noise_sigma_max"]:
                raise ValueError("Helios image noise sigma minimum cannot exceed its maximum.")
            if sigma_values["video_noise_sigma_min"] > sigma_values["video_noise_sigma_max"]:
                raise ValueError("Helios video noise sigma minimum cannot exceed its maximum.")
            call_kwargs.update(sigma_values)
        state = pipeline(**call_kwargs)
        return {
            "state_out": _issue_workflow_state(
                token=token,
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                completed_stage=self.stage,
                state=state,
            )
        }


class WorkflowDecodeVideo(_OfficialWorkflowBlockMixin, NodeBase):
    """Run Helios' official video decoder over sealed denoised state."""

    label = "Modular Workflow Decode Video"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "decode"
    action_key = "workflow_video_decoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "HeliosModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2video", "hidden": True},
        "block_path": {"type": "string", "default": "decode", "hidden": True},
        "video": {"label": "Video", "display": "output", "type": "video"},
    }

    def execute(self, **kwargs):
        pipeline_class = kwargs.get("pipeline_class")
        workflow_id = kwargs.get("workflow_id")
        _require_helios_pipeline(pipeline_class)
        token, pipeline = self._prepare_pipeline(
            pipeline_components=kwargs.get("pipeline_components"),
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            block_path=kwargs.get("block_path"),
        )
        previous_stage = _reviewed_preceding_stage(pipeline_class, workflow_id, self.action_key)
        state = _require_workflow_state(
            kwargs.get("state_in"),
            token=token,
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            completed_stage=previous_stage,
        )
        state = pipeline(state=state, output_type="pil")
        videos = state.get("videos")
        if (
            not isinstance(videos, list)
            or len(videos) != 1
            or not isinstance(videos[0], list)
            or not videos[0]
            or not all(isinstance(frame, PILImage.Image) for frame in videos[0])
        ):
            raise ValueError("The official Helios decoder did not return one PIL-frame video.")
        return {"video": videos[0]}


def _require_hunyuan_video_15_pipeline(pipeline_class, workflow_id):
    if pipeline_class != _HUNYUAN_VIDEO_15_PIPELINE or workflow_id not in _HUNYUAN_VIDEO_15_WORKFLOWS:
        raise ValueError(
            "The official HunyuanVideo 1.5 workflow node requires the exact reviewed Modular pipeline and workflow."
        )


def _hunyuan_video_15_dimensions(width, height):
    for label, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or not 256 <= value <= 2048 or value % 16:
            raise ValueError(f"HunyuanVideo 1.5 {label} must be a multiple of 16 from 256 through 2048.")


def _hunyuan_video_15_state(node, kwargs):
    pipeline_class = kwargs.get("pipeline_class")
    workflow_id = kwargs.get("workflow_id")
    _require_hunyuan_video_15_pipeline(pipeline_class, workflow_id)
    token, pipeline = node._prepare_pipeline(
        pipeline_components=kwargs.get("pipeline_components"),
        pipeline_class=pipeline_class,
        workflow_id=workflow_id,
        block_path=kwargs.get("block_path"),
    )
    previous_stage = _reviewed_preceding_stage(pipeline_class, workflow_id, node.action_key)
    state = _require_workflow_state(
        kwargs.get("state_in"),
        token=token,
        pipeline_class=pipeline_class,
        workflow_id=workflow_id,
        completed_stage=previous_stage,
    )
    return pipeline_class, workflow_id, token, pipeline, state


def _hunyuan_video_15_state_result(*, workflow_id, token, stage, state):
    return {
        "state_out": _issue_workflow_state(
            token=token,
            pipeline_class=_HUNYUAN_VIDEO_15_PIPELINE,
            workflow_id=workflow_id,
            completed_stage=stage,
            state=state,
        )
    }


class WorkflowHunyuanVideo15TextEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run HunyuanVideo 1.5's official dual text-encoder block."""

    label = "HunyuanVideo 1.5 Text Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "text_encoder"
    action_key = "workflow_hunyuan_video15_text_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "HunyuanVideo15ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2video", "hidden": True},
        "block_path": {"type": "string", "default": "text_encoder", "hidden": True},
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "negative_prompt": {
            "label": "Negative Prompt",
            "display": "textarea",
            "type": "text",
            "default": "",
        },
        "num_videos_per_prompt": {
            "label": "Videos per Prompt",
            "type": "int",
            "default": 1,
            "min": 1,
            "max": 1,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        pipeline_class = kwargs.get("pipeline_class")
        workflow_id = kwargs.get("workflow_id")
        _require_hunyuan_video_15_pipeline(pipeline_class, workflow_id)
        token, pipeline = self._prepare_pipeline(
            pipeline_components=kwargs.get("pipeline_components"),
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            block_path=kwargs.get("block_path"),
        )
        prompt = kwargs.get("prompt")
        negative_prompt = kwargs.get("negative_prompt", "")
        if type(prompt) is not str or not prompt.strip():
            raise ValueError("HunyuanVideo 1.5 requires a nonblank prompt.")
        if type(negative_prompt) is not str:
            raise ValueError("HunyuanVideo 1.5 negative prompt must be text.")
        num_videos = _bounded_integer(
            kwargs.get("num_videos_per_prompt", 1),
            label="HunyuanVideo 1.5 videos per prompt",
            minimum=1,
            maximum=1,
        )
        state = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos,
        )
        return _hunyuan_video_15_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowHunyuanVideo15VaeEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run HunyuanVideo 1.5's official I2V VAE image encoder."""

    label = "HunyuanVideo 1.5 VAE Image Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "vae_encoder"
    action_key = "workflow_hunyuan_video15_vae_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "HunyuanVideo15ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "image2video", "hidden": True},
        "block_path": {"type": "string", "default": "vae_encoder", "hidden": True},
        "image": {"label": "Reference Image", "display": "input", "type": "image", "required": True},
        # The official I2V VAE encoder derives geometry from the source image
        # when neither target dimension is supplied. Keep both controls
        # nullable so materialized workflows preserve that upstream behavior.
        "width": {"label": "Width", "type": "int", "default": None, "min": 256, "max": 2048, "step": 16},
        "height": {"label": "Height", "type": "int", "default": None, "min": 256, "max": 2048, "step": 16},
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        _pipeline_class, workflow_id, token, pipeline, state = _hunyuan_video_15_state(self, kwargs)
        if workflow_id != "image2video":
            raise ValueError("HunyuanVideo 1.5 VAE image encoding is available only for image-to-video.")
        image = kwargs.get("image")
        if not isinstance(image, PILImage.Image):
            raise ValueError("HunyuanVideo 1.5 image-to-video requires one exact PIL image.")
        width = kwargs.get("width")
        height = kwargs.get("height")
        if (width is None) != (height is None):
            raise ValueError(
                "HunyuanVideo 1.5 image-to-video target width and height must either both be omitted or both be set."
            )
        call_kwargs = {"state": state, "image": image}
        if width is not None:
            _hunyuan_video_15_dimensions(width, height)
            call_kwargs.update(width=width, height=height)
        state = pipeline(**call_kwargs)
        return _hunyuan_video_15_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowHunyuanVideo15ImageEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run HunyuanVideo 1.5's official SigLIP image encoder."""

    label = "HunyuanVideo 1.5 SigLIP Image Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "image_encoder"
    action_key = "workflow_hunyuan_video15_image_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "HunyuanVideo15ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "image2video", "hidden": True},
        "block_path": {"type": "string", "default": "image_encoder", "hidden": True},
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        _pipeline_class, workflow_id, token, pipeline, state = _hunyuan_video_15_state(self, kwargs)
        if workflow_id != "image2video":
            raise ValueError("HunyuanVideo 1.5 SigLIP image encoding is available only for image-to-video.")
        state = pipeline(state=state)
        return _hunyuan_video_15_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowHunyuanVideo15Denoise(_OfficialWorkflowBlockMixin, NodeBase):
    """Run HunyuanVideo 1.5's exact T2V or MeanFlow-aware I2V denoise block."""

    label = "HunyuanVideo 1.5 Denoise"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "denoise"
    action_key = "workflow_hunyuan_video15_denoise"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "HunyuanVideo15ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2video", "hidden": True},
        "block_path": {"type": "string", "default": "denoise", "hidden": True},
        # Geometry is an input to the official T2V denoiser. The I2V route
        # receives source-derived geometry in the state issued by vae_encoder,
        # so these shared-node controls must not silently inject T2V defaults.
        "width": {"label": "Width", "type": "int", "default": None, "min": 256, "max": 2048, "step": 16},
        "height": {"label": "Height", "type": "int", "default": None, "min": 256, "max": 2048, "step": 16},
        "num_frames": {"label": "Frames", "type": "int", "default": 121, "min": 5, "max": 481, "step": 4},
        "num_inference_steps": {
            "label": "Steps",
            "display": "slider",
            "type": "int",
            "default": 50,
            "min": 1,
            "max": 100,
        },
        "guidance_scale": {
            "label": "Guidance Scale",
            "display": "slider",
            "type": "float",
            "default": 6.0,
            "min": 1.0,
            "max": 20.0,
            "step": 0.1,
        },
        "seed": {
            "label": "Seed",
            "display": "random",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 4294967295,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        _pipeline_class, workflow_id, token, pipeline, state = _hunyuan_video_15_state(self, kwargs)
        width = kwargs.get("width")
        height = kwargs.get("height")
        if workflow_id == "text2video":
            width = 848 if width is None else width
            height = 480 if height is None else height
            _hunyuan_video_15_dimensions(width, height)
        elif width is not None or height is not None:
            raise ValueError(
                "HunyuanVideo 1.5 image-to-video geometry is owned by its VAE encoder and cannot be set on denoise."
            )
        num_frames = _bounded_integer(
            kwargs.get("num_frames", 121),
            label="HunyuanVideo 1.5 frame count",
            minimum=5,
            maximum=481,
        )
        if (num_frames - 1) % 4:
            raise ValueError("HunyuanVideo 1.5 frame count must equal 4n + 1.")
        default_steps = 12 if workflow_id == "image2video" else 50
        steps = _bounded_integer(
            kwargs.get("num_inference_steps", default_steps),
            label="HunyuanVideo 1.5 inference steps",
            minimum=1,
            maximum=100,
        )
        default_guidance = 1.0 if workflow_id == "image2video" else 6.0
        guidance_scale = float(kwargs.get("guidance_scale", default_guidance))
        if not math.isfinite(guidance_scale) or not 1.0 <= guidance_scale <= 20.0:
            raise ValueError("HunyuanVideo 1.5 guidance scale must be finite and from 1 through 20.")
        if workflow_id == "image2video" and guidance_scale != 1.0:
            raise ValueError(
                "HunyuanVideo 1.5 step-distilled image-to-video uses the reviewed fixed guidance scale of 1."
            )
        if getattr(pipeline, "guider", None) is None or not callable(getattr(pipeline.guider, "new", None)):
            raise ValueError("The official HunyuanVideo 1.5 denoise block is missing its reviewed guider component.")
        pipeline.update_components(guider=pipeline.guider.new(guidance_scale=guidance_scale))
        call_kwargs = dict(
            state=state,
            num_frames=num_frames,
            num_inference_steps=steps,
            generator=modular_generator_from_seed(kwargs.get("seed", 0), pipeline),
        )
        if workflow_id == "text2video":
            call_kwargs.update(width=width, height=height)
        state = pipeline(**call_kwargs)
        return _hunyuan_video_15_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowHunyuanVideo15Decode(_OfficialWorkflowBlockMixin, NodeBase):
    """Decode HunyuanVideo 1.5 latents into one PIL-frame video."""

    label = "HunyuanVideo 1.5 Decode Video"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "decode"
    action_key = "workflow_hunyuan_video15_decoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "HunyuanVideo15ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2video", "hidden": True},
        "block_path": {"type": "string", "default": "decode", "hidden": True},
        "video": {"label": "Video", "display": "output", "type": "video"},
    }

    def execute(self, **kwargs):
        _pipeline_class, _workflow_id, _token, pipeline, state = _hunyuan_video_15_state(self, kwargs)
        state = pipeline(state=state, output_type="pil")
        videos = state.get("videos")
        if (
            not isinstance(videos, list)
            or len(videos) != 1
            or not isinstance(videos[0], list)
            or not videos[0]
            or not all(isinstance(frame, PILImage.Image) for frame in videos[0])
        ):
            raise ValueError("The official HunyuanVideo 1.5 decoder did not return one PIL-frame video.")
        return {"video": videos[0]}


def _require_stable_diffusion_3_pipeline(pipeline_class, workflow_id):
    if pipeline_class != _STABLE_DIFFUSION_3_PIPELINE or workflow_id not in _STABLE_DIFFUSION_3_WORKFLOWS:
        raise ValueError(
            "The official Stable Diffusion 3 workflow node requires the exact reviewed Modular pipeline and workflow."
        )


def _stable_diffusion_3_dimensions(width, height):
    for label, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or not 256 <= value <= 2048 or value % 16:
            raise ValueError(f"Stable Diffusion 3 {label} must be a multiple of 16 from 256 through 2048.")


def _stable_diffusion_3_state(node, kwargs):
    pipeline_class = kwargs.get("pipeline_class")
    workflow_id = kwargs.get("workflow_id")
    _require_stable_diffusion_3_pipeline(pipeline_class, workflow_id)
    token, pipeline = node._prepare_pipeline(
        pipeline_components=kwargs.get("pipeline_components"),
        pipeline_class=pipeline_class,
        workflow_id=workflow_id,
        block_path=kwargs.get("block_path"),
    )
    state = _require_workflow_state(
        kwargs.get("state_in"),
        token=token,
        pipeline_class=pipeline_class,
        workflow_id=workflow_id,
        completed_stage=_reviewed_preceding_stage(pipeline_class, workflow_id, node.action_key),
    )
    return workflow_id, token, pipeline, state


def _stable_diffusion_3_state_result(*, workflow_id, token, stage, state):
    return {
        "state_out": _issue_workflow_state(
            token=token,
            pipeline_class=_STABLE_DIFFUSION_3_PIPELINE,
            workflow_id=workflow_id,
            completed_stage=stage,
            state=state,
        )
    }


class WorkflowStableDiffusion3TextEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run Stable Diffusion 3's official three-encoder prompt block."""

    label = "Stable Diffusion 3 Text Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "text_encoder"
    action_key = "workflow_stable_diffusion3_text_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "StableDiffusion3ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2image", "hidden": True},
        "block_path": {"type": "string", "default": "text_encoder", "hidden": True},
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "prompt_2": {"label": "CLIP 2 Prompt", "display": "textarea", "type": "text", "default": ""},
        "prompt_3": {"label": "T5 Prompt", "display": "textarea", "type": "text", "default": ""},
        "negative_prompt": {
            "label": "Negative Prompt",
            "display": "textarea",
            "type": "text",
            "default": "",
        },
        "negative_prompt_2": {
            "label": "CLIP 2 Negative Prompt",
            "display": "textarea",
            "type": "text",
            "default": "",
        },
        "negative_prompt_3": {
            "label": "T5 Negative Prompt",
            "display": "textarea",
            "type": "text",
            "default": "",
        },
        "clip_skip": {"label": "CLIP Skip", "type": "int", "default": 0, "min": 0, "max": 12},
        "max_sequence_length": {
            "label": "Maximum Sequence Length",
            "type": "int",
            "default": 256,
            "min": 1,
            "max": 512,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        pipeline_class = kwargs.get("pipeline_class")
        workflow_id = kwargs.get("workflow_id")
        _require_stable_diffusion_3_pipeline(pipeline_class, workflow_id)
        token, pipeline = self._prepare_pipeline(
            pipeline_components=kwargs.get("pipeline_components"),
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            block_path=kwargs.get("block_path"),
        )
        prompts = {
            name: kwargs.get(name, "")
            for name in (
                "prompt",
                "prompt_2",
                "prompt_3",
                "negative_prompt",
                "negative_prompt_2",
                "negative_prompt_3",
            )
        }
        if type(prompts["prompt"]) is not str or not prompts["prompt"].strip():
            raise ValueError("Stable Diffusion 3 requires a nonblank prompt.")
        if any(type(value) is not str for value in prompts.values()):
            raise ValueError("Stable Diffusion 3 prompts must be text.")
        max_sequence_length = _bounded_integer(
            kwargs.get("max_sequence_length", 256),
            label="Stable Diffusion 3 maximum sequence length",
            minimum=1,
            maximum=512,
        )
        clip_skip = _bounded_integer(
            kwargs.get("clip_skip", 0),
            label="Stable Diffusion 3 CLIP skip",
            minimum=0,
            maximum=12,
        )
        state = pipeline(
            prompt=prompts["prompt"],
            prompt_2=prompts["prompt_2"] or None,
            prompt_3=prompts["prompt_3"] or None,
            negative_prompt=prompts["negative_prompt"] or None,
            negative_prompt_2=prompts["negative_prompt_2"] or None,
            negative_prompt_3=prompts["negative_prompt_3"] or None,
            clip_skip=clip_skip or None,
            max_sequence_length=max_sequence_length,
        )
        return _stable_diffusion_3_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowStableDiffusion3VaeEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run Stable Diffusion 3's official image preprocess and VAE block."""

    label = "Stable Diffusion 3 VAE Image Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "vae_encoder"
    action_key = "workflow_stable_diffusion3_vae_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "StableDiffusion3ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "image2image", "hidden": True},
        "block_path": {"type": "string", "default": "vae_encoder", "hidden": True},
        "image": {"label": "Reference Image", "display": "input", "type": "image", "required": True},
        "width": {"label": "Width", "type": "int", "default": 1024, "min": 256, "max": 2048, "step": 16},
        "height": {"label": "Height", "type": "int", "default": 1024, "min": 256, "max": 2048, "step": 16},
        "seed": {"label": "Seed", "display": "random", "type": "int", "default": 0, "min": 0, "max": 4294967295},
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        workflow_id, token, pipeline, state = _stable_diffusion_3_state(self, kwargs)
        if workflow_id != "image2image":
            raise ValueError("Stable Diffusion 3 VAE image encoding is available only for image-to-image.")
        image = kwargs.get("image")
        if not isinstance(image, PILImage.Image):
            raise ValueError("Stable Diffusion 3 image-to-image requires one exact PIL image.")
        width = kwargs.get("width", 1024)
        height = kwargs.get("height", 1024)
        _stable_diffusion_3_dimensions(width, height)
        state = pipeline(
            state=state,
            image=image,
            width=width,
            height=height,
            generator=modular_generator_from_seed(kwargs.get("seed", 0), pipeline),
        )
        return _stable_diffusion_3_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowStableDiffusion3Denoise(_OfficialWorkflowBlockMixin, NodeBase):
    """Run Stable Diffusion 3's official T2I or I2I denoise block."""

    label = "Stable Diffusion 3 Denoise"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "denoise"
    action_key = "workflow_stable_diffusion3_denoise"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "StableDiffusion3ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2image", "hidden": True},
        "block_path": {"type": "string", "default": "denoise", "hidden": True},
        "width": {"label": "Width", "type": "int", "default": 1024, "min": 256, "max": 2048, "step": 16},
        "height": {"label": "Height", "type": "int", "default": 1024, "min": 256, "max": 2048, "step": 16},
        "num_images_per_prompt": {"label": "Images per Prompt", "type": "int", "default": 1, "min": 1, "max": 4},
        "num_inference_steps": {
            "label": "Steps",
            "display": "slider",
            "type": "int",
            "default": 50,
            "min": 1,
            "max": 100,
        },
        "guidance_scale": {
            "label": "Guidance Scale",
            "display": "slider",
            "type": "float",
            "default": 7.0,
            "min": 1.0,
            "max": 20.0,
            "step": 0.1,
        },
        "strength": {
            "label": "Strength",
            "display": "slider",
            "type": "float",
            "default": 0.6,
            "min": 0.01,
            "max": 1.0,
            "step": 0.01,
        },
        "seed": {"label": "Seed", "display": "random", "type": "int", "default": 0, "min": 0, "max": 4294967295},
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        workflow_id, token, pipeline, state = _stable_diffusion_3_state(self, kwargs)
        width = kwargs.get("width", 1024)
        height = kwargs.get("height", 1024)
        _stable_diffusion_3_dimensions(width, height)
        steps = _bounded_integer(
            kwargs.get("num_inference_steps", 50),
            label="Stable Diffusion 3 inference steps",
            minimum=1,
            maximum=100,
        )
        images_per_prompt = _bounded_integer(
            kwargs.get("num_images_per_prompt", 1),
            label="Stable Diffusion 3 images per prompt",
            minimum=1,
            maximum=4,
        )
        guidance_scale = float(kwargs.get("guidance_scale", 7.0))
        if not math.isfinite(guidance_scale) or not 1.0 <= guidance_scale <= 20.0:
            raise ValueError("Stable Diffusion 3 guidance scale must be finite and from 1 through 20.")
        if getattr(pipeline, "guider", None) is None or not callable(getattr(pipeline.guider, "new", None)):
            raise ValueError("The official Stable Diffusion 3 denoise block is missing its reviewed guider component.")
        pipeline.update_components(guider=pipeline.guider.new(guidance_scale=guidance_scale))
        call_kwargs = {
            "state": state,
            "width": width,
            "height": height,
            "num_images_per_prompt": images_per_prompt,
            "num_inference_steps": steps,
            "generator": modular_generator_from_seed(kwargs.get("seed", 0), pipeline),
        }
        if workflow_id == "image2image":
            strength = float(kwargs.get("strength", 0.6))
            if not math.isfinite(strength) or not 0.0 < strength <= 1.0:
                raise ValueError("Stable Diffusion 3 image-to-image strength must be greater than 0 and at most 1.")
            call_kwargs["strength"] = strength
        state = pipeline(**call_kwargs)
        return _stable_diffusion_3_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowStableDiffusion3Decode(_OfficialWorkflowBlockMixin, NodeBase):
    """Decode Stable Diffusion 3 latents into PIL images."""

    label = "Stable Diffusion 3 Decode Image"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "decode"
    action_key = "workflow_stable_diffusion3_decoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "StableDiffusion3ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2image", "hidden": True},
        "block_path": {"type": "string", "default": "decode", "hidden": True},
        "images": {"label": "Images", "display": "output", "type": "image"},
    }

    def execute(self, **kwargs):
        _workflow_id, _token, pipeline, state = _stable_diffusion_3_state(self, kwargs)
        state = pipeline(state=state, output_type="pil")
        images = state.get("images")
        if (
            not isinstance(images, list)
            or not images
            or not all(isinstance(image, PILImage.Image) for image in images)
        ):
            raise ValueError("The official Stable Diffusion 3 decoder did not return PIL images.")
        return {"images": images}


def _require_krea_2_pipeline(pipeline_class, workflow_id):
    if pipeline_class not in _KREA_2_PIPELINES or workflow_id != _KREA_2_WORKFLOW:
        raise ValueError("The official Krea 2 workflow node requires an exact reviewed base or Turbo pipeline.")


def _krea_2_dimensions(width, height):
    for label, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or not 256 <= value <= 2048 or value % 16:
            raise ValueError(f"Krea 2 {label} must be a multiple of 16 from 256 through 2048.")


def _krea_2_state(node, kwargs):
    pipeline_class = kwargs.get("pipeline_class")
    workflow_id = kwargs.get("workflow_id")
    _require_krea_2_pipeline(pipeline_class, workflow_id)
    token, pipeline = node._prepare_pipeline(
        pipeline_components=kwargs.get("pipeline_components"),
        pipeline_class=pipeline_class,
        workflow_id=workflow_id,
        block_path=kwargs.get("block_path"),
    )
    state = _require_workflow_state(
        kwargs.get("state_in"),
        token=token,
        pipeline_class=pipeline_class,
        workflow_id=workflow_id,
        completed_stage=_reviewed_preceding_stage(pipeline_class, workflow_id, node.action_key),
    )
    return pipeline_class, token, pipeline, state


def _krea_2_state_result(*, pipeline_class, token, stage, state):
    return {
        "state_out": _issue_workflow_state(
            token=token,
            pipeline_class=pipeline_class,
            workflow_id=_KREA_2_WORKFLOW,
            completed_stage=stage,
            state=state,
        )
    }


class WorkflowKrea2TextEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run Krea 2's official Qwen3-VL text encoder with symmetric CFG."""

    label = "Krea 2 Text Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "text_encoder"
    action_key = "workflow_krea2_text_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "Krea2ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2image", "hidden": True},
        "block_path": {"type": "string", "default": "text_encoder", "hidden": True},
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "negative_prompt": {
            "label": "Negative Prompt",
            "display": "textarea",
            "type": "text",
            "default": "",
        },
        "max_sequence_length": {
            "label": "Maximum Sequence Length",
            "type": "int",
            "default": 512,
            "min": 1,
            "max": 1024,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        pipeline_class = kwargs.get("pipeline_class")
        workflow_id = kwargs.get("workflow_id")
        if pipeline_class != _KREA_2_PIPELINE or workflow_id != _KREA_2_WORKFLOW:
            raise ValueError("The Krea 2 CFG text node requires the exact reviewed base pipeline.")
        token, pipeline = self._prepare_pipeline(
            pipeline_components=kwargs.get("pipeline_components"),
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            block_path=kwargs.get("block_path"),
        )
        prompt = kwargs.get("prompt")
        negative_prompt = kwargs.get("negative_prompt", "")
        if type(prompt) is not str or not prompt.strip():
            raise ValueError("Krea 2 requires a nonblank prompt.")
        if type(negative_prompt) is not str:
            raise ValueError("Krea 2 negative prompt must be text.")
        state = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            max_sequence_length=_bounded_integer(
                kwargs.get("max_sequence_length", 512),
                label="Krea 2 maximum sequence length",
                minimum=1,
                maximum=1024,
            ),
        )
        return _krea_2_state_result(
            pipeline_class=pipeline_class,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowKrea2TurboTextEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run Krea 2 Turbo's official guidance-free Qwen3-VL text encoder."""

    label = "Krea 2 Turbo Text Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "text_encoder"
    action_key = "workflow_krea2_turbo_text_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "Krea2TurboModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2image", "hidden": True},
        "block_path": {"type": "string", "default": "text_encoder", "hidden": True},
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "max_sequence_length": {
            "label": "Maximum Sequence Length",
            "type": "int",
            "default": 512,
            "min": 1,
            "max": 1024,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        pipeline_class = kwargs.get("pipeline_class")
        workflow_id = kwargs.get("workflow_id")
        if pipeline_class != _KREA_2_TURBO_PIPELINE or workflow_id != _KREA_2_WORKFLOW:
            raise ValueError("The Krea 2 Turbo text node requires the exact reviewed Turbo pipeline.")
        token, pipeline = self._prepare_pipeline(
            pipeline_components=kwargs.get("pipeline_components"),
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            block_path=kwargs.get("block_path"),
        )
        prompt = kwargs.get("prompt")
        if type(prompt) is not str or not prompt.strip():
            raise ValueError("Krea 2 Turbo requires a nonblank prompt.")
        state = pipeline(
            prompt=prompt,
            max_sequence_length=_bounded_integer(
                kwargs.get("max_sequence_length", 512),
                label="Krea 2 Turbo maximum sequence length",
                minimum=1,
                maximum=1024,
            ),
        )
        return _krea_2_state_result(
            pipeline_class=pipeline_class,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowKrea2Denoise(_OfficialWorkflowBlockMixin, NodeBase):
    """Run Krea 2's official symmetric-CFG denoise block."""

    label = "Krea 2 Denoise"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "denoise"
    action_key = "workflow_krea2_denoise"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "Krea2ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2image", "hidden": True},
        "block_path": {"type": "string", "default": "denoise", "hidden": True},
        "width": {"label": "Width", "type": "int", "default": 1024, "min": 256, "max": 2048, "step": 16},
        "height": {"label": "Height", "type": "int", "default": 1024, "min": 256, "max": 2048, "step": 16},
        "num_images_per_prompt": {"label": "Images per Prompt", "type": "int", "default": 1, "min": 1, "max": 4},
        "num_inference_steps": {"label": "Steps", "type": "int", "default": 28, "min": 1, "max": 100},
        "guidance_scale": {
            "label": "Guidance Scale",
            "type": "float",
            "default": 4.5,
            "min": 0.0,
            "max": 20.0,
            "step": 0.1,
        },
        "seed": {"label": "Seed", "display": "random", "type": "int", "default": 0, "min": 0, "max": 4294967295},
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        pipeline_class, token, pipeline, state = _krea_2_state(self, kwargs)
        if pipeline_class != _KREA_2_PIPELINE:
            raise ValueError("The Krea 2 CFG denoise node requires the exact reviewed base pipeline.")
        width = kwargs.get("width", 1024)
        height = kwargs.get("height", 1024)
        _krea_2_dimensions(width, height)
        guidance_scale = float(kwargs.get("guidance_scale", 4.5))
        if not math.isfinite(guidance_scale) or not 0.0 <= guidance_scale <= 20.0:
            raise ValueError("Krea 2 guidance scale must be finite and from 0 through 20.")
        if getattr(pipeline, "guider", None) is None or not callable(getattr(pipeline.guider, "new", None)):
            raise ValueError("The official Krea 2 denoise block is missing its reviewed guider component.")
        pipeline.update_components(guider=pipeline.guider.new(guidance_scale=guidance_scale))
        state = pipeline(
            state=state,
            width=width,
            height=height,
            num_images_per_prompt=_bounded_integer(
                kwargs.get("num_images_per_prompt", 1),
                label="Krea 2 images per prompt",
                minimum=1,
                maximum=4,
            ),
            num_inference_steps=_bounded_integer(
                kwargs.get("num_inference_steps", 28),
                label="Krea 2 inference steps",
                minimum=1,
                maximum=100,
            ),
            generator=modular_generator_from_seed(kwargs.get("seed", 0), pipeline),
        )
        return _krea_2_state_result(
            pipeline_class=pipeline_class,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowKrea2TurboDenoise(_OfficialWorkflowBlockMixin, NodeBase):
    """Run Krea 2 Turbo's official eight-step guidance-free denoise block."""

    label = "Krea 2 Turbo Denoise"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "denoise"
    action_key = "workflow_krea2_turbo_denoise"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "Krea2TurboModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2image", "hidden": True},
        "block_path": {"type": "string", "default": "denoise", "hidden": True},
        "width": {"label": "Width", "type": "int", "default": 1024, "min": 256, "max": 2048, "step": 16},
        "height": {"label": "Height", "type": "int", "default": 1024, "min": 256, "max": 2048, "step": 16},
        "num_images_per_prompt": {"label": "Images per Prompt", "type": "int", "default": 1, "min": 1, "max": 4},
        "num_inference_steps": {"label": "Steps", "type": "int", "default": 8, "min": 1, "max": 100},
        "seed": {"label": "Seed", "display": "random", "type": "int", "default": 0, "min": 0, "max": 4294967295},
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        pipeline_class, token, pipeline, state = _krea_2_state(self, kwargs)
        if pipeline_class != _KREA_2_TURBO_PIPELINE:
            raise ValueError("The Krea 2 Turbo denoise node requires the exact reviewed Turbo pipeline.")
        width = kwargs.get("width", 1024)
        height = kwargs.get("height", 1024)
        _krea_2_dimensions(width, height)
        state = pipeline(
            state=state,
            width=width,
            height=height,
            num_images_per_prompt=_bounded_integer(
                kwargs.get("num_images_per_prompt", 1),
                label="Krea 2 Turbo images per prompt",
                minimum=1,
                maximum=4,
            ),
            num_inference_steps=_bounded_integer(
                kwargs.get("num_inference_steps", 8),
                label="Krea 2 Turbo inference steps",
                minimum=1,
                maximum=100,
            ),
            generator=modular_generator_from_seed(kwargs.get("seed", 0), pipeline),
        )
        return _krea_2_state_result(
            pipeline_class=pipeline_class,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowKrea2Decode(_OfficialWorkflowBlockMixin, NodeBase):
    """Decode Krea 2 base or Turbo packed latents into PIL images."""

    label = "Krea 2 Decode Image"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "decode"
    action_key = "workflow_krea2_decoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "Krea2ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2image", "hidden": True},
        "block_path": {"type": "string", "default": "decode", "hidden": True},
        "images": {"label": "Images", "display": "output", "type": "image"},
    }

    def execute(self, **kwargs):
        _pipeline_class, _token, pipeline, state = _krea_2_state(self, kwargs)
        state = pipeline(state=state, output_type="pil")
        images = state.get("images")
        if (
            not isinstance(images, list)
            or not images
            or not all(isinstance(image, PILImage.Image) for image in images)
        ):
            raise ValueError("The official Krea 2 decoder did not return PIL images.")
        return {"images": images}


def _require_ideogram_4_pipeline(pipeline_class, workflow_id):
    if pipeline_class != _IDEOGRAM_4_PIPELINE or workflow_id != _IDEOGRAM_4_WORKFLOW:
        raise ValueError("The official Ideogram 4 workflow node requires the exact reviewed Modular pipeline.")


def _ideogram_4_dimensions(width, height):
    for label, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or not 256 <= value <= 4096 or value % 16:
            raise ValueError(f"Ideogram 4 {label} must be a multiple of 16 from 256 through 4096.")


def _ideogram_4_state(node, kwargs):
    pipeline_class = kwargs.get("pipeline_class")
    workflow_id = kwargs.get("workflow_id")
    _require_ideogram_4_pipeline(pipeline_class, workflow_id)
    token, pipeline = node._prepare_pipeline(
        pipeline_components=kwargs.get("pipeline_components"),
        pipeline_class=pipeline_class,
        workflow_id=workflow_id,
        block_path=kwargs.get("block_path"),
    )
    state = _require_workflow_state(
        kwargs.get("state_in"),
        token=token,
        pipeline_class=pipeline_class,
        workflow_id=workflow_id,
        completed_stage=_reviewed_preceding_stage(pipeline_class, workflow_id, node.action_key),
    )
    return token, pipeline, state


def _ideogram_4_state_result(*, token, stage, state):
    return {
        "state_out": _issue_workflow_state(
            token=token,
            pipeline_class=_IDEOGRAM_4_PIPELINE,
            workflow_id=_IDEOGRAM_4_WORKFLOW,
            completed_stage=stage,
            state=state,
        )
    }


class WorkflowIdeogram4PromptUpsample(_OfficialWorkflowBlockMixin, NodeBase):
    """Run Ideogram 4's optional local prompt-enhancer block."""

    label = "Ideogram 4 Prompt Upsample"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "prompt_upsample"
    action_key = "workflow_ideogram4_prompt_upsample"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "Ideogram4ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2image", "hidden": True},
        "block_path": {"type": "string", "default": "prompt_upsample", "hidden": True},
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "prompt_upsampling": {"label": "Local Prompt Upsampling", "type": "boolean", "default": False},
        "prompt_upsampling_temperature": {
            "label": "Upsampling Temperature",
            "type": "float",
            "default": 1.0,
            "min": 0.01,
            "max": 5.0,
            "step": 0.01,
        },
        "width": {"label": "Width", "type": "int", "default": 2048, "min": 256, "max": 4096, "step": 16},
        "height": {"label": "Height", "type": "int", "default": 2048, "min": 256, "max": 4096, "step": 16},
        "max_sequence_length": {
            "label": "Maximum Sequence Length",
            "type": "int",
            "default": 2048,
            "min": 1,
            "max": 4096,
        },
        "seed": {"label": "Seed", "display": "random", "type": "int", "default": 0, "min": 0, "max": 4294967295},
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        pipeline_class = kwargs.get("pipeline_class")
        workflow_id = kwargs.get("workflow_id")
        _require_ideogram_4_pipeline(pipeline_class, workflow_id)
        token, pipeline = self._prepare_pipeline(
            pipeline_components=kwargs.get("pipeline_components"),
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            block_path=kwargs.get("block_path"),
        )
        prompt = kwargs.get("prompt")
        if type(prompt) is not str or not prompt.strip():
            raise ValueError("Ideogram 4 requires a nonblank prompt.")
        prompt_upsampling = kwargs.get("prompt_upsampling", False)
        if type(prompt_upsampling) is not bool:
            raise ValueError("Ideogram 4 local prompt upsampling must be a boolean.")
        temperature = float(kwargs.get("prompt_upsampling_temperature", 1.0))
        if not math.isfinite(temperature) or not 0.01 <= temperature <= 5.0:
            raise ValueError("Ideogram 4 prompt upsampling temperature must be finite and from 0.01 through 5.")
        width = kwargs.get("width", 2048)
        height = kwargs.get("height", 2048)
        _ideogram_4_dimensions(width, height)
        state = pipeline(
            prompt=prompt,
            prompt_upsampling=prompt_upsampling,
            prompt_upsampling_temperature=temperature,
            width=width,
            height=height,
            max_sequence_length=_bounded_integer(
                kwargs.get("max_sequence_length", 2048),
                label="Ideogram 4 maximum sequence length",
                minimum=1,
                maximum=4096,
            ),
            generator=modular_generator_from_seed(kwargs.get("seed", 0), pipeline),
        )
        return _ideogram_4_state_result(token=token, stage=self.stage, state=state)


class WorkflowIdeogram4TextEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run Ideogram 4's official Qwen3-VL text encoder block."""

    label = "Ideogram 4 Text Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "text_encoder"
    action_key = "workflow_ideogram4_text_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "Ideogram4ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2image", "hidden": True},
        "block_path": {"type": "string", "default": "text_encoder", "hidden": True},
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        token, pipeline, state = _ideogram_4_state(self, kwargs)
        state = pipeline(state=state)
        return _ideogram_4_state_result(token=token, stage=self.stage, state=state)


class WorkflowIdeogram4Denoise(_OfficialWorkflowBlockMixin, NodeBase):
    """Run Ideogram 4's official asymmetric-CFG denoise block."""

    label = "Ideogram 4 Denoise"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "denoise"
    action_key = "workflow_ideogram4_denoise"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "Ideogram4ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2image", "hidden": True},
        "block_path": {"type": "string", "default": "denoise", "hidden": True},
        "width": {"label": "Width", "type": "int", "default": 2048, "min": 256, "max": 4096, "step": 16},
        "height": {"label": "Height", "type": "int", "default": 2048, "min": 256, "max": 4096, "step": 16},
        "num_images_per_prompt": {"label": "Images per Prompt", "type": "int", "default": 1, "min": 1, "max": 4},
        "num_inference_steps": {"label": "Steps", "type": "int", "default": 48, "min": 1, "max": 100},
        "mu": {"label": "Schedule Mu", "type": "float", "default": 0.0, "min": -10.0, "max": 10.0},
        "std": {"label": "Schedule Std", "type": "float", "default": 1.5, "min": 0.01, "max": 10.0},
        "guidance_schedule_json": {
            "label": "Guidance Schedule JSON",
            "display": "textarea",
            "type": "text",
            "default": "",
        },
        "seed": {"label": "Seed", "display": "random", "type": "int", "default": 0, "min": 0, "max": 4294967295},
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        token, pipeline, state = _ideogram_4_state(self, kwargs)
        width = kwargs.get("width", 2048)
        height = kwargs.get("height", 2048)
        _ideogram_4_dimensions(width, height)
        steps = _bounded_integer(
            kwargs.get("num_inference_steps", 48),
            label="Ideogram 4 inference steps",
            minimum=1,
            maximum=100,
        )
        mu = float(kwargs.get("mu", 0.0))
        std = float(kwargs.get("std", 1.5))
        if not math.isfinite(mu) or not -10.0 <= mu <= 10.0:
            raise ValueError("Ideogram 4 schedule mu must be finite and from -10 through 10.")
        if not math.isfinite(std) or not 0.01 <= std <= 10.0:
            raise ValueError("Ideogram 4 schedule std must be finite and from 0.01 through 10.")
        schedule_json = kwargs.get("guidance_schedule_json", "")
        if type(schedule_json) is not str:
            raise ValueError("Ideogram 4 guidance schedule JSON must be text.")
        guidance_schedule = None
        if schedule_json.strip():
            try:
                guidance_schedule = json.loads(schedule_json)
            except json.JSONDecodeError as error:
                raise ValueError("Ideogram 4 guidance schedule must be a valid JSON array.") from error
            if (
                not isinstance(guidance_schedule, list)
                or len(guidance_schedule) != steps
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
                    for value in guidance_schedule
                )
            ):
                raise ValueError(
                    "Ideogram 4 guidance schedule must contain one finite numeric value per inference step."
                )
            guidance_schedule = [float(value) for value in guidance_schedule]
        call_kwargs = {
            "state": state,
            "width": width,
            "height": height,
            "num_images_per_prompt": _bounded_integer(
                kwargs.get("num_images_per_prompt", 1),
                label="Ideogram 4 images per prompt",
                minimum=1,
                maximum=4,
            ),
            "num_inference_steps": steps,
            "mu": mu,
            "std": std,
            "generator": modular_generator_from_seed(kwargs.get("seed", 0), pipeline),
        }
        if guidance_schedule is not None:
            call_kwargs["guidance_schedule"] = guidance_schedule
        state = pipeline(**call_kwargs)
        return _ideogram_4_state_result(token=token, stage=self.stage, state=state)


class WorkflowIdeogram4Decode(_OfficialWorkflowBlockMixin, NodeBase):
    """Decode Ideogram 4 latents into PIL images."""

    label = "Ideogram 4 Decode Image"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "decode"
    action_key = "workflow_ideogram4_decoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "Ideogram4ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2image", "hidden": True},
        "block_path": {"type": "string", "default": "decode", "hidden": True},
        "images": {"label": "Images", "display": "output", "type": "image"},
    }

    def execute(self, **kwargs):
        _token, pipeline, state = _ideogram_4_state(self, kwargs)
        state = pipeline(state=state, output_type="pil")
        images = state.get("images")
        if (
            not isinstance(images, list)
            or not images
            or not all(isinstance(image, PILImage.Image) for image in images)
        ):
            raise ValueError("The official Ideogram 4 decoder did not return PIL images.")
        return {"images": images}


def _require_cosmos3_distilled_pipeline(pipeline_class, workflow_id):
    if pipeline_class != _COSMOS3_DISTILLED_PIPELINE or workflow_id not in _COSMOS3_DISTILLED_WORKFLOWS:
        raise ValueError(
            "The official Cosmos 3 Distilled workflow node requires the exact reviewed Modular pipeline and workflow."
        )


def _cosmos3_distilled_dimensions(width, height):
    for label, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or not 256 <= value <= 2048 or value % 16:
            raise ValueError(f"Cosmos 3 Distilled {label} must be a multiple of 16 from 256 through 2048.")


def _cosmos3_distilled_state(node, kwargs):
    pipeline_class = kwargs.get("pipeline_class")
    workflow_id = kwargs.get("workflow_id")
    _require_cosmos3_distilled_pipeline(pipeline_class, workflow_id)
    token, pipeline = node._prepare_pipeline(
        pipeline_components=kwargs.get("pipeline_components"),
        pipeline_class=pipeline_class,
        workflow_id=workflow_id,
        block_path=kwargs.get("block_path"),
    )
    state = _require_workflow_state(
        kwargs.get("state_in"),
        token=token,
        pipeline_class=pipeline_class,
        workflow_id=workflow_id,
        completed_stage=_reviewed_preceding_stage(pipeline_class, workflow_id, node.action_key),
    )
    return workflow_id, token, pipeline, state


def _cosmos3_distilled_state_result(*, workflow_id, token, stage, state):
    return {
        "state_out": _issue_workflow_state(
            token=token,
            pipeline_class=_COSMOS3_DISTILLED_PIPELINE,
            workflow_id=workflow_id,
            completed_stage=stage,
            state=state,
        )
    }


class WorkflowCosmos3DistilledTextEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run Cosmos 3 Distilled's official tokenizer and prompt-template block."""

    label = "Cosmos 3 Distilled Text Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "text_encoder"
    action_key = "workflow_cosmos3_distilled_text_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "Cosmos3DistilledModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2video", "hidden": True},
        "block_path": {"type": "string", "default": "text_encoder", "hidden": True},
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "width": {"label": "Width", "type": "int", "default": 1280, "min": 256, "max": 2048, "step": 16},
        "height": {"label": "Height", "type": "int", "default": 720, "min": 256, "max": 2048, "step": 16},
        "num_frames": {"label": "Frames", "type": "int", "default": 189, "min": 1, "max": 481},
        "fps": {"label": "FPS", "type": "float", "default": 24.0, "min": 1.0, "max": 120.0},
        "use_system_prompt": {"label": "Use System Prompt", "type": "boolean", "default": True},
        "add_resolution_template": {"label": "Add Resolution Template", "type": "boolean", "default": True},
        "add_duration_template": {"label": "Add Duration Template", "type": "boolean", "default": True},
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        pipeline_class = kwargs.get("pipeline_class")
        workflow_id = kwargs.get("workflow_id")
        _require_cosmos3_distilled_pipeline(pipeline_class, workflow_id)
        token, pipeline = self._prepare_pipeline(
            pipeline_components=kwargs.get("pipeline_components"),
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            block_path=kwargs.get("block_path"),
        )
        prompt = kwargs.get("prompt")
        if type(prompt) is not str or not prompt.strip():
            raise ValueError("Cosmos 3 Distilled requires a nonblank prompt.")
        width = kwargs.get("width", 1280)
        height = kwargs.get("height", 720)
        _cosmos3_distilled_dimensions(width, height)
        num_frames = _bounded_integer(
            kwargs.get("num_frames", 1 if workflow_id == "text2image" else 189),
            label="Cosmos 3 Distilled frame count",
            minimum=1,
            maximum=481,
        )
        if workflow_id == "text2image" and num_frames != 1:
            raise ValueError("Cosmos 3 Distilled text-to-image requires exactly one frame.")
        if workflow_id != "text2image" and num_frames < 2:
            raise ValueError("Cosmos 3 Distilled video workflows require at least two frames.")
        fps = float(kwargs.get("fps", 24.0))
        if not math.isfinite(fps) or not 1.0 <= fps <= 120.0:
            raise ValueError("Cosmos 3 Distilled FPS must be finite and from 1 through 120.")
        template_flags = {}
        for field in ("use_system_prompt", "add_resolution_template", "add_duration_template"):
            value = kwargs.get(field, True)
            if type(value) is not bool:
                raise ValueError(f"Cosmos 3 Distilled {field} must be a boolean.")
            template_flags[field] = value
        state = pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            **template_flags,
        )
        return _cosmos3_distilled_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowCosmos3DistilledVaeEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run Cosmos 3 Distilled's exact conditional image or video VAE block."""

    label = "Cosmos 3 Distilled Condition Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "vae_encoder"
    action_key = "workflow_cosmos3_distilled_vae_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "Cosmos3DistilledModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "image2video", "hidden": True},
        "block_path": {"type": "string", "default": "vae_encoder", "hidden": True},
        "image": {"label": "Reference Image", "display": "input", "type": "image"},
        "video": {"label": "Reference Video", "display": "input", "type": "video"},
        "condition_frame_indexes_vision": {
            "label": "Condition Latent Frames",
            "display": "textarea",
            "type": "text",
            "default": "[0, 1]",
        },
        "condition_video_keep": {
            "label": "Condition Video Keep",
            "type": "string",
            "default": "first",
            "options": ["first", "last"],
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        workflow_id, token, pipeline, state = _cosmos3_distilled_state(self, kwargs)
        if workflow_id == "image2video":
            image = kwargs.get("image")
            if not isinstance(image, PILImage.Image):
                raise ValueError("Cosmos 3 Distilled image-to-video requires one exact PIL image.")
            call_kwargs = {"state": state, "image": image}
        elif workflow_id == "video2video":
            video = kwargs.get("video")
            if (
                not isinstance(video, list)
                or not video
                or not all(isinstance(frame, PILImage.Image) for frame in video)
            ):
                raise ValueError("Cosmos 3 Distilled video-to-video requires decoded PIL frames.")
            raw_indexes = kwargs.get("condition_frame_indexes_vision", "[0, 1]")
            if type(raw_indexes) is str:
                try:
                    indexes = json.loads(raw_indexes)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Cosmos 3 Distilled condition latent frames must be a JSON integer list."
                    ) from error
            else:
                indexes = raw_indexes
            if (
                not isinstance(indexes, list)
                or not indexes
                or not all(type(index) is int and index >= 0 for index in indexes)
            ):
                raise ValueError("Cosmos 3 Distilled condition latent frames must be a nonempty integer list.")
            keep = kwargs.get("condition_video_keep", "first")
            if keep not in {"first", "last"}:
                raise ValueError("Cosmos 3 Distilled condition video keep must be 'first' or 'last'.")
            call_kwargs = {
                "state": state,
                "video": video,
                "condition_frame_indexes_vision": indexes,
                "condition_video_keep": keep,
            }
        else:
            raise ValueError("Cosmos 3 Distilled VAE conditioning exists only in image/video-conditioned workflows.")
        state = pipeline(**call_kwargs)
        return _cosmos3_distilled_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowCosmos3DistilledDenoise(_OfficialWorkflowBlockMixin, NodeBase):
    """Run Cosmos 3 Distilled's official fixed-schedule vision denoise block."""

    label = "Cosmos 3 Distilled Denoise"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "denoise"
    action_key = "workflow_cosmos3_distilled_denoise"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "Cosmos3DistilledModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2video", "hidden": True},
        "block_path": {"type": "string", "default": "denoise", "hidden": True},
        "num_inference_steps": {"label": "Distilled Steps", "type": "int", "default": 4, "min": 1, "max": 16},
        "guidance_scale": {"label": "Guidance Scale", "type": "float", "default": 1.0, "min": 1.0, "max": 1.0},
        "seed": {
            "label": "Seed",
            "display": "random",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 4294967295,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        workflow_id, token, pipeline, state = _cosmos3_distilled_state(self, kwargs)
        steps = _bounded_integer(
            kwargs.get("num_inference_steps", 4),
            label="Cosmos 3 Distilled inference steps",
            minimum=1,
            maximum=16,
        )
        if steps != 4:
            raise ValueError(
                "Cosmos 3 Distilled requires the official fixed four-step schedule from distilled_sigmas."
            )
        guidance_scale = float(kwargs.get("guidance_scale", 1.0))
        if guidance_scale != 1.0:
            raise ValueError("Cosmos 3 Distilled uses the official fixed guidance scale of 1.")
        state = pipeline(
            state=state,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=modular_generator_from_seed(kwargs.get("seed", 0), pipeline),
        )
        return _cosmos3_distilled_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowCosmos3DistilledDecode(_OfficialWorkflowBlockMixin, NodeBase):
    """Decode Cosmos 3 Distilled vision latents to one image or PIL-frame video."""

    label = "Cosmos 3 Distilled Decode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "decode"
    action_key = "workflow_cosmos3_distilled_decoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "Cosmos3DistilledModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2video", "hidden": True},
        "block_path": {"type": "string", "default": "decode", "hidden": True},
        "image": {"label": "Image", "display": "output", "type": "image"},
        "video": {"label": "Video", "display": "output", "type": "video"},
    }

    def execute(self, **kwargs):
        workflow_id, _token, pipeline, state = _cosmos3_distilled_state(self, kwargs)
        state = pipeline(state=state, output_type="pil")
        frames = state.get("videos")
        if (
            not isinstance(frames, list)
            or not frames
            or not all(isinstance(frame, PILImage.Image) for frame in frames)
        ):
            raise ValueError("The official Cosmos 3 Distilled decoder did not return PIL frames.")
        if workflow_id == "text2image":
            if len(frames) != 1:
                raise ValueError("Cosmos 3 Distilled text-to-image must decode exactly one frame.")
            return {"image": frames[0]}
        return {"video": frames}


def _require_cosmos3_omni_pipeline(pipeline_class, workflow_id):
    if pipeline_class != _COSMOS3_OMNI_PIPELINE or workflow_id not in _COSMOS3_OMNI_WORKFLOWS:
        raise ValueError(
            "The official Cosmos 3 Omni workflow node requires the exact reviewed Modular pipeline and workflow."
        )


def _cosmos3_omni_state(node, kwargs):
    pipeline_class = kwargs.get("pipeline_class")
    workflow_id = kwargs.get("workflow_id")
    _require_cosmos3_omni_pipeline(pipeline_class, workflow_id)
    token, pipeline = node._prepare_pipeline(
        pipeline_components=kwargs.get("pipeline_components"),
        pipeline_class=pipeline_class,
        workflow_id=workflow_id,
        block_path=kwargs.get("block_path"),
    )
    state = _require_workflow_state(
        kwargs.get("state_in"),
        token=token,
        pipeline_class=pipeline_class,
        workflow_id=workflow_id,
        completed_stage=_reviewed_preceding_stage(pipeline_class, workflow_id, node.action_key),
    )
    return workflow_id, token, pipeline, state


def _cosmos3_omni_state_result(*, workflow_id, token, stage, state):
    return {
        "state_out": _issue_workflow_state(
            token=token,
            pipeline_class=_COSMOS3_OMNI_PIPELINE,
            workflow_id=workflow_id,
            completed_stage=stage,
            state=state,
        )
    }


def _cosmos3_condition_indexes(raw_indexes):
    if type(raw_indexes) is str:
        try:
            indexes = json.loads(raw_indexes)
        except json.JSONDecodeError as error:
            raise ValueError("Cosmos 3 condition latent frames must be a JSON integer list.") from error
    else:
        indexes = raw_indexes
    if not isinstance(indexes, list) or not indexes or not all(type(index) is int and index >= 0 for index in indexes):
        raise ValueError("Cosmos 3 condition latent frames must be a nonempty integer list.")
    return indexes


class WorkflowCosmos3OmniTextEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run Cosmos 3 Omni's exact standard or action-aware text block."""

    label = "Cosmos 3 Omni Text Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "text_encoder"
    action_key = "workflow_cosmos3_omni_text_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "Cosmos3OmniModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2video", "hidden": True},
        "block_path": {"type": "string", "default": "text_encoder", "hidden": True},
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "negative_prompt": {
            "label": "Negative Prompt",
            "display": "textarea",
            "type": "text",
            "default": "",
        },
        "image": {"label": "Action Reference Image", "display": "input", "type": "image"},
        "video": {"label": "Action Reference Video", "display": "input", "type": "video"},
        "width": {"label": "Width", "type": "int", "default": 1280, "min": 256, "max": 2048, "step": 16},
        "height": {"label": "Height", "type": "int", "default": 720, "min": 256, "max": 2048, "step": 16},
        "num_frames": {"label": "Frames", "type": "int", "default": 189, "min": 1, "max": 481},
        "fps": {"label": "FPS", "type": "float", "default": 24.0, "min": 1.0, "max": 120.0},
        "use_system_prompt": {"label": "Use System Prompt", "type": "boolean", "default": True},
        "add_resolution_template": {"label": "Add Resolution Template", "type": "boolean", "default": True},
        "add_duration_template": {"label": "Add Duration Template", "type": "boolean", "default": True},
        "action_chunk_size": {"label": "Action Chunk Size", "type": "int", "default": 16, "min": 1, "max": 256},
        "action_domain_name": {"label": "Action Domain", "type": "string", "default": "droid_lerobot"},
        "action_resolution_tier": {
            "label": "Action Resolution Tier",
            "type": "int",
            "default": 480,
            "options": [256, 480, 704, 720],
        },
        "action_view_point": {
            "label": "Action Viewpoint",
            "type": "string",
            "default": "ego_view",
            "options": ["ego_view", "third_person_view", "wrist_view", "concat_view"],
        },
        "raw_actions_json": {
            "label": "Raw Actions (JSON)",
            "display": "textarea",
            "type": "text",
            "default": "[]",
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def _action_condition(self, workflow_id, kwargs):
        from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import CosmosActionCondition

        mode = _COSMOS3_OMNI_ACTION_MODES[workflow_id]
        image = kwargs.get("image")
        video = kwargs.get("video")
        if mode == "inverse_dynamics":
            if (
                not isinstance(video, list)
                or not video
                or not all(isinstance(frame, PILImage.Image) for frame in video)
            ):
                raise ValueError("Cosmos 3 inverse dynamics requires decoded PIL reference frames.")
            image = None
        else:
            if not isinstance(image, PILImage.Image):
                raise ValueError("Cosmos 3 policy and forward dynamics require one exact PIL reference image.")
            video = None
        raw_actions = None
        if mode == "forward_dynamics":
            try:
                values = json.loads(kwargs.get("raw_actions_json", "[]"))
            except json.JSONDecodeError as error:
                raise ValueError("Cosmos 3 raw actions must be a JSON matrix.") from error
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(row, list) and row for row in values)
                or not all(
                    type(value) in {int, float} and math.isfinite(float(value)) for row in values for value in row
                )
            ):
                raise ValueError("Cosmos 3 forward-dynamics raw actions must be a nonempty finite JSON matrix.")
            row_widths = {len(row) for row in values}
            if len(row_widths) != 1:
                raise ValueError("Cosmos 3 raw-action rows must have one consistent width.")
            import torch

            raw_actions = torch.tensor(values, dtype=torch.float32)
        return CosmosActionCondition(
            mode=mode,
            chunk_size=_bounded_integer(
                kwargs.get("action_chunk_size", 16),
                label="Cosmos 3 action chunk size",
                minimum=1,
                maximum=256,
            ),
            domain_name=kwargs.get("action_domain_name", "droid_lerobot"),
            resolution_tier=kwargs.get("action_resolution_tier", 480),
            raw_actions=raw_actions,
            image=image,
            video=video,
            view_point=kwargs.get("action_view_point", "ego_view"),
        )

    def execute(self, **kwargs):
        pipeline_class = kwargs.get("pipeline_class")
        workflow_id = kwargs.get("workflow_id")
        _require_cosmos3_omni_pipeline(pipeline_class, workflow_id)
        token, pipeline = self._prepare_pipeline(
            pipeline_components=kwargs.get("pipeline_components"),
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            block_path=kwargs.get("block_path"),
        )
        prompt = kwargs.get("prompt")
        negative_prompt = kwargs.get("negative_prompt", "")
        if type(prompt) is not str or not prompt.strip():
            raise ValueError("Cosmos 3 Omni requires a nonblank prompt.")
        if type(negative_prompt) is not str:
            raise ValueError("Cosmos 3 Omni negative prompt must be text.")
        fps = float(kwargs.get("fps", 24.0))
        if not math.isfinite(fps) or not 1.0 <= fps <= 120.0:
            raise ValueError("Cosmos 3 Omni FPS must be finite and from 1 through 120.")
        template_flags = {}
        for field in ("use_system_prompt", "add_resolution_template", "add_duration_template"):
            value = kwargs.get(field, True)
            if type(value) is not bool:
                raise ValueError(f"Cosmos 3 Omni {field} must be a boolean.")
            template_flags[field] = value
        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "fps": fps,
            **template_flags,
        }
        if workflow_id in _COSMOS3_OMNI_ACTION_MODES:
            call_kwargs["action"] = self._action_condition(workflow_id, kwargs)
        else:
            width = kwargs.get("width", 1280)
            height = kwargs.get("height", 720)
            _cosmos3_distilled_dimensions(width, height)
            num_frames = _bounded_integer(
                kwargs.get("num_frames", 1 if workflow_id == "text2image" else 189),
                label="Cosmos 3 Omni frame count",
                minimum=1,
                maximum=481,
            )
            if workflow_id == "text2image" and num_frames != 1:
                raise ValueError("Cosmos 3 Omni text-to-image requires exactly one frame.")
            if workflow_id != "text2image" and num_frames < 2:
                raise ValueError("Cosmos 3 Omni video workflows require at least two frames.")
            call_kwargs.update(width=width, height=height, num_frames=num_frames)
        state = pipeline(**call_kwargs)
        return _cosmos3_omni_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowCosmos3OmniVaeEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run Cosmos 3 Omni's exact action/image/video conditional VAE block."""

    label = "Cosmos 3 Omni Condition Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "vae_encoder"
    action_key = "workflow_cosmos3_omni_vae_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "Cosmos3OmniModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "image2video", "hidden": True},
        "block_path": {"type": "string", "default": "vae_encoder", "hidden": True},
        "image": {"label": "Reference Image", "display": "input", "type": "image"},
        "video": {"label": "Reference Video", "display": "input", "type": "video"},
        "condition_frame_indexes_vision": {
            "label": "Condition Latent Frames",
            "display": "textarea",
            "type": "text",
            "default": "[0, 1]",
        },
        "condition_video_keep": {
            "label": "Condition Video Keep",
            "type": "string",
            "default": "first",
            "options": ["first", "last"],
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        workflow_id, token, pipeline, state = _cosmos3_omni_state(self, kwargs)
        call_kwargs = {"state": state}
        if workflow_id in {"image2video", "image2video_with_sound"}:
            image = kwargs.get("image")
            if not isinstance(image, PILImage.Image):
                raise ValueError("Cosmos 3 Omni image-to-video requires one exact PIL image.")
            call_kwargs["image"] = image
        elif workflow_id in {"video2video", "video2video_with_sound"}:
            video = kwargs.get("video")
            if (
                not isinstance(video, list)
                or not video
                or not all(isinstance(frame, PILImage.Image) for frame in video)
            ):
                raise ValueError("Cosmos 3 Omni video-to-video requires decoded PIL frames.")
            keep = kwargs.get("condition_video_keep", "first")
            if keep not in {"first", "last"}:
                raise ValueError("Cosmos 3 Omni condition video keep must be 'first' or 'last'.")
            call_kwargs.update(
                video=video,
                condition_frame_indexes_vision=_cosmos3_condition_indexes(
                    kwargs.get("condition_frame_indexes_vision", "[0, 1]")
                ),
                condition_video_keep=keep,
            )
        elif workflow_id not in _COSMOS3_OMNI_ACTION_MODES:
            raise ValueError("Cosmos 3 Omni VAE conditioning is not part of this reviewed workflow.")
        state = pipeline(**call_kwargs)
        return _cosmos3_omni_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowCosmos3OmniDenoise(_OfficialWorkflowBlockMixin, NodeBase):
    """Run Cosmos 3 Omni's exact modality-selected denoise block."""

    label = "Cosmos 3 Omni Denoise"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "denoise"
    action_key = "workflow_cosmos3_omni_denoise"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "Cosmos3OmniModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2video", "hidden": True},
        "block_path": {"type": "string", "default": "denoise", "hidden": True},
        "num_inference_steps": {"label": "Steps", "type": "int", "default": 50, "min": 1, "max": 100},
        "guidance_scale": {
            "label": "Guidance Scale",
            "type": "float",
            "default": 6.0,
            "min": 1.0,
            "max": 20.0,
            "step": 0.1,
        },
        "seed": {
            "label": "Seed",
            "display": "random",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 4294967295,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        workflow_id, token, pipeline, state = _cosmos3_omni_state(self, kwargs)
        steps = _bounded_integer(
            kwargs.get("num_inference_steps", 50),
            label="Cosmos 3 Omni inference steps",
            minimum=1,
            maximum=100,
        )
        guidance_scale = float(kwargs.get("guidance_scale", 6.0))
        if not math.isfinite(guidance_scale) or not 1.0 <= guidance_scale <= 20.0:
            raise ValueError("Cosmos 3 Omni guidance scale must be finite and from 1 through 20.")
        state = pipeline(
            state=state,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            enable_sound=workflow_id in _COSMOS3_OMNI_SOUND_WORKFLOWS,
            generator=modular_generator_from_seed(kwargs.get("seed", 0), pipeline),
        )
        return _cosmos3_omni_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowCosmos3OmniDecode(_OfficialWorkflowBlockMixin, NodeBase):
    """Decode Cosmos 3 Omni vision and optional synchronized sound latents."""

    label = "Cosmos 3 Omni Decode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "decode"
    action_key = "workflow_cosmos3_omni_decoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "Cosmos3OmniModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2video", "hidden": True},
        "block_path": {"type": "string", "default": "decode", "hidden": True},
        "image": {"label": "Image", "display": "output", "type": "image"},
        # Preserve the official Modular state/output name.  The one-frame
        # text2image route is additionally projected through ``image``.
        "videos": {"label": "Videos", "display": "output", "type": "video"},
        "audio": {"label": "Audio", "display": "output", "type": "audio"},
        "sample_rate": {"label": "Sample Rate", "display": "output", "type": "int"},
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        workflow_id, token, pipeline, state = _cosmos3_omni_state(self, kwargs)
        state = pipeline(state=state, output_type="pil")
        frames = state.get("videos")
        if (
            not isinstance(frames, list)
            or not frames
            or not all(isinstance(frame, PILImage.Image) for frame in frames)
        ):
            raise ValueError("The official Cosmos 3 Omni decoder did not return PIL frames.")
        result = _cosmos3_omni_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )
        if workflow_id == "text2image":
            if len(frames) != 1:
                raise ValueError("Cosmos 3 Omni text-to-image must decode exactly one frame.")
            result["image"] = frames[0]
        else:
            result["videos"] = frames
        if workflow_id in _COSMOS3_OMNI_SOUND_WORKFLOWS:
            raw_audio = state.get("sound")
            sample_rate = state.get("sampling_rate")
            if (
                raw_audio is None
                or isinstance(sample_rate, bool)
                or not isinstance(sample_rate, int)
                or sample_rate <= 0
            ):
                raise ValueError("The official Cosmos 3 Omni sound decoder did not return audio and a sample rate.")
            from modules.DiffusersAudio.main import output_to_audio_object

            result["audio"] = output_to_audio_object(raw_audio, sample_rate=sample_rate)
            result["sample_rate"] = sample_rate
        return result


class WorkflowCosmos3OmniAfterDecode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run Cosmos 3 Omni's official post-decode action-output block."""

    label = "Cosmos 3 Omni Action Output"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "after_decode"
    action_key = "workflow_cosmos3_omni_after_decode"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "Cosmos3OmniModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2video", "hidden": True},
        "block_path": {"type": "string", "default": "after_decode", "hidden": True},
        "action": {"label": "Action", "display": "output", "type": "any"},
    }

    def execute(self, **kwargs):
        _workflow_id, _token, pipeline, state = _cosmos3_omni_state(self, kwargs)
        state = pipeline(state=state)
        return {"action": state.get("action")}


def _require_minimax_h3_pipeline(pipeline_class, workflow_id):
    if pipeline_class != _MINIMAX_H3_PIPELINE or workflow_id not in _MINIMAX_H3_WORKFLOWS:
        raise ValueError(
            "The official MiniMax H3 workflow node requires the exact reviewed Modular pipeline and workflow."
        )


def _minimax_h3_dimensions(width, height):
    for label, value in (("width", width), ("height", height)):
        # The pinned upstream blocks require a pixel dimension divisible by the
        # checkpoint's canvas multiple. They do not publish an arbitrary UI
        # upper bound, so Expert mode must not invent one here.
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % 32:
            raise ValueError(f"MiniMax H3 {label} must be a positive multiple of 32.")


def _minimax_h3_reference_types():
    """Resolve the package-owned public reference dataclasses lazily."""

    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3Reference,
        MiniMaxH3VideoReference,
    )

    return (
        MiniMaxH3Reference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
        MiniMaxH3AudioReference,
    )


def _validated_minimax_h3_references(references, *, allow_audio_only):
    """Validate one ordered public-reference value without rewriting its order."""

    MiniMaxH3Reference, _image_type, _video_type, _audio_type = _minimax_h3_reference_types()
    if not isinstance(references, list):
        raise ValueError("MiniMax H3 references must be an ordered list.")
    for index, reference in enumerate(references):
        if not isinstance(reference, MiniMaxH3Reference):
            raise ValueError(
                f"MiniMax H3 reference {index} is not an official image, video, or audio reference dataclass."
            )
    kinds = [reference.kind for reference in references]
    for kind, maximum in (("image", 9), ("video", 3), ("audio", 3)):
        if kinds.count(kind) > maximum:
            raise ValueError(f"MiniMax H3 accepts at most {maximum} {kind} references.")
    if len(references) > 12:
        raise ValueError("MiniMax H3 accepts at most 12 ordered references in total.")
    if references and not allow_audio_only and set(kinds) == {"audio"}:
        raise ValueError("A MiniMax H3 audio reference must be paired with an image or video reference.")
    return list(references)


class WorkflowMiniMaxH3ReferenceAssembler(NodeBase):
    """Append one decoded medium to MiniMax H3's semantic ordered reference list.

    The node constructs the exact public Diffusers dataclass and never opens a
    path or URL. Chaining these nodes therefore preserves the user-selected
    order while keeping all file/network access in MoDiff's reviewed media
    loaders.
    """

    label = "MiniMax H3 Add Ordered Reference"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    params = {
        "references_in": {
            "label": "Existing Ordered References",
            "display": "input",
            "type": "minimax_h3_references",
        },
        "reference_kind": {
            "label": "Reference Kind",
            "type": "string",
            "default": "image",
            "options": ["image", "video", "audio"],
        },
        "image": {"label": "Image", "display": "input", "type": "image"},
        "video": {"label": "Video Frames", "display": "input", "type": "video"},
        "video_fps": {"label": "Video FPS", "type": "float", "default": 24.0, "min": 0.01},
        "audio": {"label": "Audio or Video Soundtrack", "display": "input", "type": "audio"},
        "references": {
            "label": "Ordered References",
            "display": "output",
            "type": "minimax_h3_references",
        },
    }

    @staticmethod
    def _audio_reference_payload(audio):
        if audio is None:
            return None, None
        from modules.DiffusersAudio.main import audio_to_numpy

        samples, sample_rate = audio_to_numpy(audio, clip=False)
        if (
            getattr(samples, "ndim", None) != 2
            or samples.shape[0] not in {1, 2}
            or samples.shape[1] < 1
            or not bool(samples.size)
        ):
            raise ValueError("MiniMax H3 reference audio must be a nonempty mono or stereo waveform.")
        if not bool((abs(samples) < float("inf")).all()):
            raise ValueError("MiniMax H3 reference audio samples must all be finite.")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("MiniMax H3 reference audio must carry a positive integer sample rate.")
        import torch

        return torch.from_numpy(samples.copy()).to(dtype=torch.float32), sample_rate

    def execute(self, **kwargs):
        references = _validated_minimax_h3_references(
            kwargs.get("references_in") or [],
            # An intermediate list may start with audio and be completed by a
            # later chained assembler. The executable ref2va boundary performs
            # the official no-audio-only validation.
            allow_audio_only=True,
        )
        reference_kind = kwargs.get("reference_kind", "image")
        MiniMaxH3Reference, MiniMaxH3ImageReference, MiniMaxH3VideoReference, MiniMaxH3AudioReference = (
            _minimax_h3_reference_types()
        )
        del MiniMaxH3Reference

        if reference_kind == "image":
            image = kwargs.get("image")
            if not isinstance(image, PILImage.Image):
                raise ValueError("MiniMax H3 image references require one decoded PIL image.")
            reference = MiniMaxH3ImageReference(image=image)
        elif reference_kind == "video":
            frames = kwargs.get("video")
            if (
                not isinstance(frames, list)
                or not frames
                or not all(isinstance(frame, PILImage.Image) for frame in frames)
            ):
                raise ValueError("MiniMax H3 video references require a nonempty list of decoded PIL frames.")
            fps = float(kwargs.get("video_fps", 24.0))
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("MiniMax H3 reference video FPS must be finite and positive.")
            soundtrack, sample_rate = self._audio_reference_payload(kwargs.get("audio"))
            reference = MiniMaxH3VideoReference(
                frames=list(frames),
                fps=fps,
                audio=soundtrack,
                sample_rate=sample_rate,
            )
        elif reference_kind == "audio":
            waveform, sample_rate = self._audio_reference_payload(kwargs.get("audio"))
            if waveform is None:
                raise ValueError("MiniMax H3 audio references require decoded audio.")
            reference = MiniMaxH3AudioReference(audio=waveform, sample_rate=sample_rate)
        else:
            raise ValueError("MiniMax H3 reference kind must be image, video, or audio.")

        references.append(reference)
        return {
            "references": _validated_minimax_h3_references(references, allow_audio_only=True),
        }


def _minimax_h3_state(node, kwargs):
    pipeline_class = kwargs.get("pipeline_class")
    workflow_id = kwargs.get("workflow_id")
    _require_minimax_h3_pipeline(pipeline_class, workflow_id)
    token, pipeline = node._prepare_pipeline(
        pipeline_components=kwargs.get("pipeline_components"),
        pipeline_class=pipeline_class,
        workflow_id=workflow_id,
        block_path=kwargs.get("block_path"),
    )
    state = _require_workflow_state(
        kwargs.get("state_in"),
        token=token,
        pipeline_class=pipeline_class,
        workflow_id=workflow_id,
        completed_stage=_reviewed_preceding_stage(pipeline_class, workflow_id, node.action_key),
    )
    return workflow_id, token, pipeline, state


def _minimax_h3_state_result(*, workflow_id, token, stage, state):
    return {
        "state_out": _issue_workflow_state(
            token=token,
            pipeline_class=_MINIMAX_H3_PIPELINE,
            workflow_id=workflow_id,
            completed_stage=stage,
            state=state,
        )
    }


class WorkflowMiniMaxH3BeforeEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run MiniMax H3's exact keyframe or omni-reference preparation block."""

    label = "MiniMax H3 Prepare Media"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "before_encode"
    action_key = "workflow_minimax_h3_before_encode"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "MiniMaxH3ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "fl2va", "hidden": True},
        "block_path": {"type": "string", "default": "before_encode", "hidden": True},
        "image": {"label": "First Frame", "display": "input", "type": "image"},
        "last_image": {"label": "Last Frame", "display": "input", "type": "image"},
        "references": {
            "label": "Ordered References",
            "display": "input",
            "type": "minimax_h3_references",
        },
        "width": {"label": "Width", "type": "int", "default": 1344, "min": 32, "step": 32},
        "height": {"label": "Height", "type": "int", "default": 768, "min": 32, "step": 32},
        "num_frames": {"label": "Frames", "type": "int", "default": 124, "min": 120, "max": 360},
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        pipeline_class = kwargs.get("pipeline_class")
        workflow_id = kwargs.get("workflow_id")
        _require_minimax_h3_pipeline(pipeline_class, workflow_id)
        if workflow_id == "t2va":
            raise ValueError("MiniMax H3 text-only generation has no before-encode stage.")
        token, pipeline = self._prepare_pipeline(
            pipeline_components=kwargs.get("pipeline_components"),
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            block_path=kwargs.get("block_path"),
        )
        width = kwargs.get("width", 1344)
        height = kwargs.get("height", 768)
        _minimax_h3_dimensions(width, height)
        num_frames = _bounded_integer(
            kwargs.get("num_frames", 124),
            label="MiniMax H3 frame count",
            minimum=120,
            maximum=360,
        )
        if workflow_id == "fl2va":
            image = kwargs.get("image")
            last_image = kwargs.get("last_image")
            if image is not None and not isinstance(image, PILImage.Image):
                raise ValueError("MiniMax H3 first frame must be one exact PIL image.")
            if last_image is not None and not isinstance(last_image, PILImage.Image):
                raise ValueError("MiniMax H3 last frame must be one exact PIL image.")
            if image is None and last_image is None:
                raise ValueError("MiniMax H3 first/last-frame generation requires at least one keyframe.")
            state = pipeline(
                image=image,
                last_image=last_image,
                width=width,
                height=height,
                num_frames=num_frames,
            )
        else:
            references = _validated_minimax_h3_references(
                kwargs.get("references"),
                allow_audio_only=False,
            )
            if not references:
                raise ValueError("MiniMax H3 omni-reference generation requires a nonempty ordered reference list.")
            state = pipeline(
                references=references,
                width=width,
                height=height,
                num_frames=num_frames,
            )
        return _minimax_h3_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowMiniMaxH3TextEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run MiniMax H3's exact Qwen3-VL presentation encoder."""

    label = "MiniMax H3 Text Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "text_encoder"
    action_key = "workflow_minimax_h3_text_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {"label": "Workflow State", "display": "input", "type": "modular_workflow_state"},
        "pipeline_class": {"type": "string", "default": "MiniMaxH3ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "t2va", "hidden": True},
        "block_path": {"type": "string", "default": "text_encoder", "hidden": True},
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        pipeline_class = kwargs.get("pipeline_class")
        workflow_id = kwargs.get("workflow_id")
        _require_minimax_h3_pipeline(pipeline_class, workflow_id)
        token, pipeline = self._prepare_pipeline(
            pipeline_components=kwargs.get("pipeline_components"),
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            block_path=kwargs.get("block_path"),
        )
        prompt = kwargs.get("prompt")
        if type(prompt) is not str or not prompt.strip():
            raise ValueError("MiniMax H3 requires a nonblank prompt.")
        if workflow_id == "t2va":
            state = pipeline(prompt=prompt)
        else:
            state = _require_workflow_state(
                kwargs.get("state_in"),
                token=token,
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                completed_stage="before_encode",
            )
            state = pipeline(state=state, prompt=prompt)
        return _minimax_h3_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowMiniMaxH3VaeEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run MiniMax H3's exact keyframe or reference VAE encoder."""

    label = "MiniMax H3 Condition Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "vae_encoder"
    action_key = "workflow_minimax_h3_vae_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "MiniMaxH3ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "fl2va", "hidden": True},
        "block_path": {"type": "string", "default": "vae_encoder", "hidden": True},
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        workflow_id, token, pipeline, state = _minimax_h3_state(self, kwargs)
        if workflow_id == "t2va":
            raise ValueError("MiniMax H3 text-only generation has no VAE conditioning stage.")
        state = pipeline(state=state)
        return _minimax_h3_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowMiniMaxH3Denoise(_OfficialWorkflowBlockMixin, NodeBase):
    """Run MiniMax H3's exact task-selected packed video/audio denoiser."""

    label = "MiniMax H3 Denoise Video and Audio"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "denoise"
    action_key = "workflow_minimax_h3_denoise"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "MiniMaxH3ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "t2va", "hidden": True},
        "block_path": {"type": "string", "default": "denoise", "hidden": True},
        "width": {"label": "Width", "type": "int", "default": 1344, "min": 32, "step": 32},
        "height": {"label": "Height", "type": "int", "default": 768, "min": 32, "step": 32},
        "num_frames": {"label": "Frames", "type": "int", "default": 124, "min": 120, "max": 360},
        "num_inference_steps": {"label": "Steps", "type": "int", "default": 50, "min": 1},
        "seed": {
            "label": "Seed",
            "display": "random",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 4294967295,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        workflow_id, token, pipeline, state = _minimax_h3_state(self, kwargs)
        steps = kwargs.get("num_inference_steps", 50)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("MiniMax H3 inference steps must be a positive integer.")
        call_kwargs = {
            "state": state,
            "num_inference_steps": steps,
            "generator": modular_generator_from_seed(kwargs.get("seed", 0), pipeline),
        }
        if workflow_id == "t2va":
            width = kwargs.get("width", 1344)
            height = kwargs.get("height", 768)
            _minimax_h3_dimensions(width, height)
            call_kwargs.update(
                width=width,
                height=height,
                num_frames=_bounded_integer(
                    kwargs.get("num_frames", 124),
                    label="MiniMax H3 frame count",
                    minimum=120,
                    maximum=360,
                ),
            )
        state = pipeline(**call_kwargs)
        return _minimax_h3_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowMiniMaxH3Decode(_OfficialWorkflowBlockMixin, NodeBase):
    """Decode MiniMax H3's joint video and stereo soundtrack."""

    label = "MiniMax H3 Decode Video and Audio"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "decode"
    action_key = "workflow_minimax_h3_decoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "MiniMaxH3ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "t2va", "hidden": True},
        "block_path": {"type": "string", "default": "decode", "hidden": True},
        "video": {"label": "Video", "display": "output", "type": "video"},
        "audio": {"label": "Audio", "display": "output", "type": "audio"},
        "sample_rate": {"label": "Sample Rate", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        _workflow_id, _token, pipeline, state = _minimax_h3_state(self, kwargs)
        state = pipeline(state=state, output_type="pil")
        videos = state.get("videos")
        if (
            not isinstance(videos, list)
            or len(videos) != 1
            or not isinstance(videos[0], list)
            or not videos[0]
            or not all(isinstance(frame, PILImage.Image) for frame in videos[0])
        ):
            raise ValueError("The official MiniMax H3 decoder did not return one PIL-frame video.")
        raw_audio = state.get("audio")
        sample_rate = state.get("sampling_rate")
        if raw_audio is None or isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("The official MiniMax H3 decoder did not return audio and a sample rate.")
        from modules.DiffusersAudio.main import output_to_audio_object

        return {
            "video": videos[0],
            "audio": output_to_audio_object(raw_audio, sample_rate=sample_rate),
            "sample_rate": sample_rate,
        }


def _require_ltx25_pipeline(pipeline_class, workflow_id):
    if pipeline_class != _LTX_25_PIPELINE or workflow_id not in _LTX_25_WORKFLOWS:
        raise ValueError(
            "The official LTX-2.5 workflow node requires the exact reviewed Modular pipeline and workflow."
        )


def _ltx25_dimensions(width, height):
    for label, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or not 256 <= value <= 2048 or value % 32:
            raise ValueError(f"LTX-2.5 {label} must be a multiple of 32 from 256 through 2048.")


def _ltx25_frames(value):
    frames = _bounded_integer(value, label="LTX-2.5 frame count", minimum=9, maximum=481)
    if (frames - 1) % 8:
        raise ValueError("LTX-2.5 frame count must equal 8n + 1.")
    return frames


def _ltx25_noise_scale(value):
    """Preserve the upstream route-specific meaning of an omitted noise scale."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("LTX-2.5 noise scale must be null or a finite number from 0 through 1.")
    try:
        noise_scale = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("LTX-2.5 noise scale must be null or a finite number from 0 through 1.") from error
    if not math.isfinite(noise_scale) or not 0.0 <= noise_scale <= 1.0:
        raise ValueError("LTX-2.5 noise scale must be null or a finite number from 0 through 1.")
    return noise_scale


def _ltx25_condition_classes():
    """Resolve only the two official LTX-2 condition containers at execution time."""

    try:
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
        from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import LTX2ReferenceCondition
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "LTX-2.5 condition execution requires MoDiff's reviewed Transformers optional runtime. "
            "Install and activate it through Setup before running this workflow."
        ) from error
    return LTX2VideoCondition, LTX2ReferenceCondition


def _ltx25_condition_frames(value, *, label, allow_single_batched_video=False):
    """Reject arbitrary objects before an official condition reaches Diffusers."""

    import numpy as np
    import torch

    if isinstance(value, PILImage.Image):
        return
    if type(value) is list:
        if not value or not all(isinstance(frame, PILImage.Image) for frame in value):
            raise ValueError(f"{label} frames must be a nonempty list of PIL images.")
        return
    if type(value) is np.ndarray:
        allowed_dimensions = {3, 4, 5} if allow_single_batched_video else {3, 4}
        if value.ndim not in allowed_dimensions or any(dimension <= 0 for dimension in value.shape):
            expected = "3D, 4D, or single-batch 5D" if allow_single_batched_video else "3D or 4D"
            raise ValueError(f"{label} array frames must have a nonempty {expected} shape.")
        if value.ndim == 5 and value.shape[0] != 1:
            raise ValueError(f"{label} array frames support exactly one batched video.")
        return
    if type(value) is torch.Tensor:
        allowed_dimensions = {3, 4, 5} if allow_single_batched_video else {3, 4}
        if (
            value.layout is not torch.strided
            or value.ndim not in allowed_dimensions
            or any(dimension <= 0 for dimension in value.shape)
            or value.is_complex()
        ):
            expected = "3D, 4D, or single-batch 5D" if allow_single_batched_video else "3D or 4D"
            raise ValueError(f"{label} tensor frames must be a nonempty, strided, real {expected} tensor.")
        if value.ndim == 5 and value.shape[0] != 1:
            raise ValueError(f"{label} tensor frames support exactly one batched video.")
        return
    raise TypeError(
        f"{label} frames must be a PIL image, a nonempty PIL-image list, or an exact NumPy/Torch tensor."
    )


def _ltx25_unit_interval(value, *, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number from 0 through 1.")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number from 0 through 1.") from error
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{label} must be a finite number from 0 through 1.")
    return normalized


def _ltx25_video_conditions(value, *, required):
    video_condition_class, _reference_condition_class = _ltx25_condition_classes()
    if value is None:
        if required:
            raise ValueError("LTX-2.5 condition-to-video requires at least one official video condition.")
        return None
    values = [value] if type(value) is video_condition_class else value
    if type(values) is not list or not values:
        raise ValueError("LTX-2.5 video conditions must be an official condition or a nonempty list of them.")
    for index, condition in enumerate(values):
        if type(condition) is not video_condition_class:
            raise TypeError(
                "LTX-2.5 video conditions accept only exact Diffusers LTX2VideoCondition instances."
            )
        _ltx25_condition_frames(condition.frames, label=f"LTX-2.5 video condition {index}")
        if isinstance(condition.index, bool) or not isinstance(condition.index, int):
            raise ValueError("LTX-2.5 video-condition indexes must be integers.")
        _ltx25_unit_interval(condition.strength, label="LTX-2.5 video-condition strength")
        if condition.crf is not None and (
            isinstance(condition.crf, bool)
            or not isinstance(condition.crf, int)
            or not 0 <= condition.crf <= 51
        ):
            raise ValueError("LTX-2.5 video-condition CRF must be null or an integer from 0 through 51.")
    return values


def _ltx25_reference_conditions(value, *, required):
    _video_condition_class, reference_condition_class = _ltx25_condition_classes()
    if value is None:
        if required:
            raise ValueError("LTX-2.5 in-context generation requires an official reference condition.")
        return None
    values = [value] if type(value) is reference_condition_class else value
    if type(values) is not list or not values:
        raise ValueError("LTX-2.5 reference conditions must be an official condition or a nonempty list of them.")
    for index, condition in enumerate(values):
        if type(condition) is not reference_condition_class:
            raise TypeError(
                "LTX-2.5 reference conditions accept only exact Diffusers LTX2ReferenceCondition instances."
            )
        _ltx25_condition_frames(
            condition.frames,
            label=f"LTX-2.5 reference condition {index}",
            allow_single_batched_video=True,
        )
        _ltx25_unit_interval(condition.strength, label="LTX-2.5 reference-condition strength")
    return values


def _ltx25_conditioning_attention_mask(value):
    if value is None:
        return None
    import torch

    if type(value) is not torch.Tensor:
        raise TypeError("LTX-2.5 conditioning attention mask must be an exact Torch tensor.")
    if (
        value.layout is not torch.strided
        or not value.is_floating_point()
        or value.is_complex()
        or value.ndim != 5
        or value.shape[0] != 1
        or value.shape[1] != 1
        or any(dimension <= 0 for dimension in value.shape[2:])
    ):
        raise ValueError(
            "LTX-2.5 conditioning attention mask must be a nonempty floating tensor shaped (1, 1, F, H, W)."
        )
    if not bool(torch.isfinite(value).all().item()) or float(value.amin().item()) < 0.0 or float(
        value.amax().item()
    ) > 1.0:
        raise ValueError("LTX-2.5 conditioning attention mask values must be finite and from 0 through 1.")
    return value


def _ltx25_vocoder_sample_rate(pipeline):
    """Read the exact output rate from the loaded LTX-2.5 vocoder component."""

    vocoder = getattr(pipeline, "vocoder", None)
    config = getattr(vocoder, "config", None)
    if hasattr(config, "get"):
        sample_rate = config.get("output_sampling_rate")
    else:
        sample_rate = getattr(config, "output_sampling_rate", None)
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(
            "The loaded LTX-2.5 vocoder config must declare a positive integer output_sampling_rate."
        )
    return sample_rate


def _ltx25_state(node, kwargs):
    pipeline_class = kwargs.get("pipeline_class")
    workflow_id = kwargs.get("workflow_id")
    _require_ltx25_pipeline(pipeline_class, workflow_id)
    token, pipeline = node._prepare_pipeline(
        pipeline_components=kwargs.get("pipeline_components"),
        pipeline_class=pipeline_class,
        workflow_id=workflow_id,
        block_path=kwargs.get("block_path"),
    )
    state = _require_workflow_state(
        kwargs.get("state_in"),
        token=token,
        pipeline_class=pipeline_class,
        workflow_id=workflow_id,
        completed_stage=_reviewed_preceding_stage(pipeline_class, workflow_id, node.action_key),
    )
    return workflow_id, token, pipeline, state


def _ltx25_state_result(*, workflow_id, token, stage, state):
    return {
        "state_out": _issue_workflow_state(
            token=token,
            pipeline_class=_LTX_25_PIPELINE,
            workflow_id=workflow_id,
            completed_stage=stage,
            state=state,
        )
    }


class WorkflowLTX25TextEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run LTX-2.5's exact Gemma text encoder and connector stage."""

    label = "LTX-2.5 Text Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "text_encoder"
    action_key = "workflow_ltx25_text_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "LTX25ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2video", "hidden": True},
        "block_path": {"type": "string", "default": "text_encoder", "hidden": True},
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "negative_prompt": {
            "label": "Negative Prompt",
            "display": "textarea",
            "type": "text",
            "default": "",
        },
        "max_sequence_length": {
            "label": "Max Sequence Length",
            "type": "int",
            "default": 1024,
            "min": 64,
            "max": 4096,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        pipeline_class = kwargs.get("pipeline_class")
        workflow_id = kwargs.get("workflow_id")
        _require_ltx25_pipeline(pipeline_class, workflow_id)
        token, pipeline = self._prepare_pipeline(
            pipeline_components=kwargs.get("pipeline_components"),
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            block_path=kwargs.get("block_path"),
        )
        prompt = kwargs.get("prompt")
        negative_prompt = kwargs.get("negative_prompt", "")
        if type(prompt) is not str or not prompt.strip():
            raise ValueError("LTX-2.5 requires a nonblank prompt.")
        if type(negative_prompt) is not str:
            raise ValueError("LTX-2.5 negative prompt must be text.")
        max_length = _bounded_integer(
            kwargs.get("max_sequence_length", 1024),
            label="LTX-2.5 maximum sequence length",
            minimum=64,
            maximum=4096,
        )
        state = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_sequence_length=max_length,
        )
        return _ltx25_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowLTX25Duration(_OfficialWorkflowBlockMixin, NodeBase):
    """Run or explicitly bypass LTX-2.5's official duration-head stage."""

    label = "LTX-2.5 Resolve Duration"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "duration"
    action_key = "workflow_ltx25_duration"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "LTX25ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2video", "hidden": True},
        "block_path": {"type": "string", "default": "duration", "hidden": True},
        "auto_duration": {"label": "Auto Duration", "type": "boolean", "default": True},
        "num_frames": {"label": "Fixed Frames", "type": "int", "default": 121, "min": 9, "max": 481},
        "min_seconds": {"label": "Minimum Seconds", "type": "float", "default": 1.0, "min": 0.1, "max": 20.0},
        "max_seconds": {"label": "Maximum Seconds", "type": "float", "default": 20.0, "min": 0.2, "max": 60.0},
        "frame_rate": {"label": "Frame Rate", "type": "float", "default": 24.0, "min": 1.0, "max": 120.0},
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        workflow_id, token, pipeline, state = _ltx25_state(self, kwargs)
        if workflow_id == "in_context":
            raise ValueError(
                "LTX-2.5 in-context generation requires an explicit frame count and has no duration stage."
            )
        auto_duration = kwargs.get("auto_duration", True)
        if type(auto_duration) is not bool:
            raise ValueError("LTX-2.5 auto duration must be a boolean.")
        min_seconds = float(kwargs.get("min_seconds", 1.0))
        max_seconds = float(kwargs.get("max_seconds", 20.0))
        frame_rate = float(kwargs.get("frame_rate", 24.0))
        if (
            not all(math.isfinite(value) for value in (min_seconds, max_seconds, frame_rate))
            or min_seconds <= 0
            or max_seconds <= min_seconds
            or not 1.0 <= frame_rate <= 120.0
        ):
            raise ValueError("LTX-2.5 duration bounds and frame rate are invalid.")
        state = pipeline(
            state=state,
            num_frames=None if auto_duration else _ltx25_frames(kwargs.get("num_frames", 121)),
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            frame_rate=frame_rate,
        )
        return _ltx25_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowLTX25VaeEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run LTX-2.5's official image-to-video VAE encoder."""

    label = "LTX-2.5 Image Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "vae_encoder"
    action_key = "workflow_ltx25_vae_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "LTX25ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "image2video", "hidden": True},
        "block_path": {"type": "string", "default": "vae_encoder", "hidden": True},
        "image": {"label": "Reference Image", "display": "input", "type": "image", "required": True},
        "width": {"label": "Width", "type": "int", "default": 704, "min": 256, "max": 2048, "step": 32},
        "height": {"label": "Height", "type": "int", "default": 512, "min": 256, "max": 2048, "step": 32},
        "image_crf": {"label": "Image CRF", "type": "int", "default": 18, "min": 0, "max": 51},
        "seed": {
            "label": "Seed",
            "display": "random",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 4294967295,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        workflow_id, token, pipeline, state = _ltx25_state(self, kwargs)
        if workflow_id != "image2video":
            raise ValueError("LTX-2.5 image VAE encoding exists only in image-to-video.")
        image = kwargs.get("image")
        if not isinstance(image, PILImage.Image):
            raise ValueError("LTX-2.5 image-to-video requires one exact PIL image.")
        width = kwargs.get("width", 704)
        height = kwargs.get("height", 512)
        _ltx25_dimensions(width, height)
        image_crf = _bounded_integer(
            kwargs.get("image_crf", 18),
            label="LTX-2.5 image CRF",
            minimum=0,
            maximum=51,
        )
        state = pipeline(
            state=state,
            image=image,
            width=width,
            height=height,
            image_crf=image_crf,
            generator=continuation_generator_from_seed(kwargs.get("seed", 0), pipeline, state),
        )
        return _ltx25_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowLTX25ConditionEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run LTX-2.5's official condition encoder for condition and IC-LoRA workflows."""

    label = "LTX-2.5 Condition Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "condition_encoder"
    action_key = "workflow_ltx25_condition_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "LTX25ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "condition", "hidden": True},
        "block_path": {"type": "string", "default": "condition_encoder", "hidden": True},
        "conditions": {"label": "Video Conditions", "display": "input", "type": "any"},
        "reference_conditions": {"label": "Reference Conditions", "display": "input", "type": "any"},
        "width": {"label": "Width", "type": "int", "default": 704, "min": 256, "max": 2048, "step": 32},
        "height": {"label": "Height", "type": "int", "default": 512, "min": 256, "max": 2048, "step": 32},
        "num_frames": {"label": "Frames", "type": "int", "default": 121, "min": 9, "max": 481},
        "seed": {
            "label": "Seed",
            "display": "random",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 4294967295,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        workflow_id, token, pipeline, state = _ltx25_state(self, kwargs)
        if workflow_id not in {"condition", "in_context"}:
            raise ValueError("LTX-2.5 condition encoding is not part of this reviewed workflow.")
        width = kwargs.get("width", 704)
        height = kwargs.get("height", 512)
        _ltx25_dimensions(width, height)
        generator = continuation_generator_from_seed(kwargs.get("seed", 0), pipeline, state)
        call_kwargs = {
            "state": state,
            "width": width,
            "height": height,
            "generator": generator,
        }
        if workflow_id == "condition":
            conditions = _ltx25_video_conditions(kwargs.get("conditions"), required=True)
            if kwargs.get("reference_conditions") is not None:
                raise ValueError("LTX-2.5 condition-to-video does not accept in-context references.")
            call_kwargs["conditions"] = conditions
        else:
            conditions = _ltx25_video_conditions(kwargs.get("conditions"), required=False)
            reference_conditions = _ltx25_reference_conditions(
                kwargs.get("reference_conditions"),
                required=True,
            )
            call_kwargs.update(
                conditions=conditions,
                reference_conditions=reference_conditions,
                num_frames=_ltx25_frames(kwargs.get("num_frames", 121)),
            )
        state = pipeline(**call_kwargs)
        return _ltx25_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowLTX25ReferenceEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run LTX-2.5's official in-context reference-token encoder."""

    label = "LTX-2.5 Reference Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "reference_encoder"
    action_key = "workflow_ltx25_reference_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "LTX25ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "in_context", "hidden": True},
        "block_path": {"type": "string", "default": "reference_encoder", "hidden": True},
        "reference_conditions": {"label": "Reference Conditions", "display": "input", "type": "any"},
        "reference_downscale_factor": {
            "label": "Reference Downscale",
            "type": "int",
            "default": 1,
            "min": 1,
            "max": 8,
        },
        "conditioning_attention_strength": {
            "label": "Conditioning Attention",
            "type": "float",
            "default": 1.0,
            "min": 0.0,
            "max": 1.0,
        },
        "conditioning_attention_mask": {
            "label": "Conditioning Attention Mask",
            "display": "input",
            "type": "tensor",
        },
        "frame_rate": {"label": "Frame Rate", "type": "float", "default": 24.0, "min": 1.0, "max": 120.0},
        "seed": {
            "label": "Seed",
            "display": "random",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 4294967295,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        workflow_id, token, pipeline, state = _ltx25_state(self, kwargs)
        if workflow_id != "in_context":
            raise ValueError("LTX-2.5 reference encoding exists only in the in-context workflow.")
        references = _ltx25_reference_conditions(kwargs.get("reference_conditions"), required=True)
        attention_mask = _ltx25_conditioning_attention_mask(kwargs.get("conditioning_attention_mask"))
        downscale = _bounded_integer(
            kwargs.get("reference_downscale_factor", 1),
            label="LTX-2.5 reference downscale factor",
            minimum=1,
            maximum=8,
        )
        strength = float(kwargs.get("conditioning_attention_strength", 1.0))
        frame_rate = float(kwargs.get("frame_rate", 24.0))
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("LTX-2.5 conditioning attention strength must be finite and from 0 through 1.")
        if not math.isfinite(frame_rate) or not 1.0 <= frame_rate <= 120.0:
            raise ValueError("LTX-2.5 frame rate must be finite and from 1 through 120.")
        state = pipeline(
            state=state,
            reference_conditions=references,
            reference_downscale_factor=downscale,
            conditioning_attention_strength=strength,
            conditioning_attention_mask=attention_mask,
            frame_rate=frame_rate,
            generator=continuation_generator_from_seed(kwargs.get("seed", 0), pipeline, state),
        )
        return _ltx25_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowLTX25Denoise(_OfficialWorkflowBlockMixin, NodeBase):
    """Run LTX-2.5's exact auto-selected joint video/audio denoise block."""

    label = "LTX-2.5 Denoise Video and Audio"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "denoise"
    action_key = "workflow_ltx25_denoise"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "LTX25ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2video", "hidden": True},
        "block_path": {"type": "string", "default": "denoise", "hidden": True},
        "width": {"label": "Width", "type": "int", "default": 704, "min": 256, "max": 2048, "step": 32},
        "height": {"label": "Height", "type": "int", "default": 512, "min": 256, "max": 2048, "step": 32},
        "frame_rate": {"label": "Frame Rate", "type": "float", "default": 24.0, "min": 1.0, "max": 120.0},
        "num_videos_per_prompt": {"label": "Videos per Prompt", "type": "int", "default": 1, "min": 1, "max": 4},
        # The official default is deliberately null. The selected upstream
        # workflow resolves it differently: T2V/I2V use 0.0, while
        # condition/in-context use the first custom sigma or 1.0.
        "noise_scale": {"label": "Noise Scale", "type": "float", "default": None, "min": 0.0, "max": 1.0},
        "use_cross_timestep": {"label": "Use Cross Timestep", "type": "boolean", "default": True},
        "seed": {
            "label": "Seed",
            "display": "random",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 4294967295,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        workflow_id, token, pipeline, state = _ltx25_state(self, kwargs)
        if kwargs.get("num_inference_steps") is not None:
            raise ValueError(
                "LTX-2.5 distilled execution uses its exact reviewed eight-sigma schedule and does not accept "
                "a num_inference_steps substitute."
            )
        width = kwargs.get("width", 704)
        height = kwargs.get("height", 512)
        _ltx25_dimensions(width, height)
        frame_rate = float(kwargs.get("frame_rate", 24.0))
        noise_scale = _ltx25_noise_scale(kwargs.get("noise_scale"))
        use_cross_timestep = kwargs.get("use_cross_timestep", True)
        if not math.isfinite(frame_rate) or not 1.0 <= frame_rate <= 120.0:
            raise ValueError("LTX-2.5 frame rate must be finite and from 1 through 120.")
        if type(use_cross_timestep) is not bool:
            raise ValueError("LTX-2.5 cross-timestep control must be a boolean.")
        configure_ltx25_distilled_denoise_components(pipeline)
        call_kwargs = {
            "state": state,
            "frame_rate": frame_rate,
            "num_videos_per_prompt": _bounded_integer(
                kwargs.get("num_videos_per_prompt", 1),
                label="LTX-2.5 videos per prompt",
                minimum=1,
                maximum=4,
            ),
            "noise_scale": noise_scale,
            "use_cross_timestep": use_cross_timestep,
            "generator": continuation_generator_from_seed(kwargs.get("seed", 0), pipeline, state),
            **ltx25_distilled_denoise_kwargs(),
        }
        if workflow_id == "text2video":
            call_kwargs.update(width=width, height=height)
        state = pipeline(**call_kwargs)
        return _ltx25_state_result(
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowLTX25Decode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run LTX-2.5's native diffusion video decoder and audio vocoder."""

    label = "LTX-2.5 Decode Video and Audio"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "decode"
    action_key = "workflow_ltx25_decoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "LTX25ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "text2video", "hidden": True},
        "block_path": {"type": "string", "default": "decode", "hidden": True},
        "seed": {
            "label": "Decoder Seed",
            "display": "random",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 4294967295,
        },
        "video": {"label": "Video", "display": "output", "type": "video"},
        "audio": {"label": "Audio", "display": "output", "type": "audio"},
        "sample_rate": {"label": "Sample Rate", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        _workflow_id, _token, pipeline, state = _ltx25_state(self, kwargs)
        configure_ltx25_diffusion_decoder(pipeline)
        state = pipeline(
            state=state,
            output_type="pil",
            generator=continuation_generator_from_seed(kwargs.get("seed", 0), pipeline, state),
        )
        videos = state.get("videos")
        if (
            not isinstance(videos, list)
            or len(videos) != 1
            or not isinstance(videos[0], list)
            or not videos[0]
            or not all(isinstance(frame, PILImage.Image) for frame in videos[0])
        ):
            raise ValueError("The official LTX-2.5 decoder did not return one PIL-frame video.")
        raw_audio = state.get("audio")
        if raw_audio is None:
            raise ValueError("The official LTX-2.5 decoder did not return a generated waveform.")
        from modules.DiffusersAudio.main import output_to_audio_object

        sample_rate = _ltx25_vocoder_sample_rate(pipeline)
        return {
            "video": videos[0],
            "audio": output_to_audio_object(raw_audio, sample_rate=sample_rate),
            "sample_rate": sample_rate,
        }


def _require_wan_animate_pipeline(pipeline_class, workflow_id):
    if pipeline_class not in _WAN_ANIMATE_PIPELINES or workflow_id != _WAN_ANIMATE_WORKFLOW:
        raise ValueError(
            "The official Wan Animate 2 workflow node requires an exact reviewed base or Distilled pipeline class."
        )


def _wan_animate_dimensions(width, height):
    for label, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or not 256 <= value <= 2048 or value % 16:
            raise ValueError(f"Wan Animate 2 {label} must be a multiple of 16 from 256 through 2048.")


def _wan_animate_state(node, kwargs):
    pipeline_class = kwargs.get("pipeline_class")
    workflow_id = kwargs.get("workflow_id")
    _require_wan_animate_pipeline(pipeline_class, workflow_id)
    token, pipeline = node._prepare_pipeline(
        pipeline_components=kwargs.get("pipeline_components"),
        pipeline_class=pipeline_class,
        workflow_id=workflow_id,
        block_path=kwargs.get("block_path"),
    )
    previous_stage = _reviewed_preceding_stage(pipeline_class, workflow_id, node.action_key)
    state = _require_workflow_state(
        kwargs.get("state_in"),
        token=token,
        pipeline_class=pipeline_class,
        workflow_id=workflow_id,
        completed_stage=previous_stage,
    )
    return pipeline_class, workflow_id, token, pipeline, state


def _wan_animate_state_result(*, pipeline_class, workflow_id, token, stage, state):
    return {
        "state_out": _issue_workflow_state(
            token=token,
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            completed_stage=stage,
            state=state,
        )
    }


class WorkflowWanAnimateTextEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Run Wan Animate 2's official text-encoder block."""

    label = "Wan Animate 2 Text Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "text_encoder"
    action_key = "workflow_wan_animate_text_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "WanAnimate2ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "default", "hidden": True},
        "block_path": {"type": "string", "default": "text_encoder", "hidden": True},
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "negative_prompt": {
            "label": "Negative Prompt",
            "display": "textarea",
            "type": "text",
            "default": "",
        },
        "prompt_ref": {
            "label": "Driving Video Reference Prompt",
            "display": "textarea",
            "type": "text",
            "default": "人物动作的参考视频",
        },
        "max_sequence_length": {
            "label": "Maximum Sequence Length",
            "type": "int",
            "default": 512,
            "min": 1,
            "max": 512,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        pipeline_class = kwargs.get("pipeline_class")
        workflow_id = kwargs.get("workflow_id")
        _require_wan_animate_pipeline(pipeline_class, workflow_id)
        token, pipeline = self._prepare_pipeline(
            pipeline_components=kwargs.get("pipeline_components"),
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            block_path=kwargs.get("block_path"),
        )
        prompt = kwargs.get("prompt")
        negative_prompt = kwargs.get("negative_prompt", "")
        prompt_ref = kwargs.get("prompt_ref", "人物动作的参考视频")
        if type(prompt) is not str or not prompt.strip():
            raise ValueError("Wan Animate 2 requires a nonblank prompt.")
        if type(negative_prompt) is not str:
            raise ValueError("Wan Animate 2 negative prompt must be text.")
        if type(prompt_ref) is not str or not prompt_ref.strip():
            raise ValueError("Wan Animate 2 driving-video reference prompt must be nonblank text.")
        max_sequence_length = _bounded_integer(
            kwargs.get("max_sequence_length", 512),
            label="Wan Animate 2 maximum sequence length",
            minimum=1,
            maximum=512,
        )
        state = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_ref=prompt_ref,
            max_sequence_length=max_sequence_length,
        )
        return _wan_animate_state_result(
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowWanAnimateImageEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Preprocess and CLIP-encode Wan Animate 2's reference image."""

    label = "Wan Animate 2 Reference Image Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "image_encoder"
    action_key = "workflow_wan_animate_image_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "WanAnimate2ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "default", "hidden": True},
        "block_path": {"type": "string", "default": "image_encoder", "hidden": True},
        "image": {"label": "Reference Image", "display": "input", "type": "image", "required": True},
        "width": {"label": "Target Area Width", "type": "int", "default": 640, "min": 256, "max": 2048, "step": 16},
        "height": {"label": "Target Area Height", "type": "int", "default": 800, "min": 256, "max": 2048, "step": 16},
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        pipeline_class, workflow_id, token, pipeline, state = _wan_animate_state(self, kwargs)
        image = kwargs.get("image")
        if not isinstance(image, PILImage.Image):
            raise ValueError("Wan Animate 2 requires one exact PIL reference image.")
        width = kwargs.get("width", 640)
        height = kwargs.get("height", 800)
        _wan_animate_dimensions(width, height)
        state = pipeline(state=state, image=image, width=width, height=height)
        return _wan_animate_state_result(
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowWanAnimateVideoEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """Preprocess and CLIP-encode Wan Animate 2's driving video."""

    label = "Wan Animate 2 Driving Video Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "video_encoder"
    action_key = "workflow_wan_animate_video_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "WanAnimate2ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "default", "hidden": True},
        "block_path": {"type": "string", "default": "video_encoder", "hidden": True},
        "driving_video": {"label": "Driving Video", "display": "input", "type": "video", "required": True},
        "driving_video_fps": {"label": "Driving Video FPS", "display": "input", "type": "float", "required": True},
        "fps": {"label": "Output FPS", "type": "int", "default": 24, "min": 1, "max": 120},
        "segment_frame_length": {
            "label": "Segment Frame Length",
            "type": "int",
            "default": 81,
            "min": 5,
            "max": 481,
            "step": 4,
        },
        "prev_segment_conditioning_frames": {
            "label": "Previous Segment Conditioning Frames",
            "type": "int",
            "default": 1,
            "min": 0,
            "max": 80,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        pipeline_class, workflow_id, token, pipeline, state = _wan_animate_state(self, kwargs)
        video = kwargs.get("driving_video")
        if not isinstance(video, list) or not video or not all(isinstance(frame, PILImage.Image) for frame in video):
            raise ValueError("Wan Animate 2 requires a nonempty driving video decoded to PIL frames.")
        driving_video_fps = float(kwargs.get("driving_video_fps", 0.0))
        if not math.isfinite(driving_video_fps) or driving_video_fps <= 0.0:
            raise ValueError("Wan Animate 2 driving-video FPS must be finite and greater than zero.")
        fps = _bounded_integer(kwargs.get("fps", 24), label="Wan Animate 2 output FPS", minimum=1, maximum=120)
        segment_frame_length = _bounded_integer(
            kwargs.get("segment_frame_length", 81),
            label="Wan Animate 2 segment frame length",
            minimum=5,
            maximum=481,
        )
        if (segment_frame_length - 1) % 4:
            raise ValueError("Wan Animate 2 segment frame length must equal 4n + 1.")
        previous_frames = _bounded_integer(
            kwargs.get("prev_segment_conditioning_frames", 1),
            label="Wan Animate 2 previous-segment conditioning frames",
            minimum=0,
            maximum=80,
        )
        if previous_frames >= segment_frame_length:
            raise ValueError("Wan Animate 2 previous-segment conditioning frames must be less than segment length.")
        state = pipeline(
            state=state,
            driving_video=video,
            driving_video_fps=driving_video_fps,
            fps=fps,
            segment_frame_length=segment_frame_length,
            prev_segment_conditioning_frames=previous_frames,
        )
        return _wan_animate_state_result(
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowWanAnimateVaeEncode(_OfficialWorkflowBlockMixin, NodeBase):
    """VAE-encode Wan Animate 2's preprocessed reference image."""

    label = "Wan Animate 2 Reference VAE Encode"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "vae_encoder"
    action_key = "workflow_wan_animate_vae_encoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "WanAnimate2ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "default", "hidden": True},
        "block_path": {"type": "string", "default": "vae_encoder", "hidden": True},
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        pipeline_class, workflow_id, token, pipeline, state = _wan_animate_state(self, kwargs)
        state = pipeline(state=state)
        return _wan_animate_state_result(
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowWanAnimateDenoise(_OfficialWorkflowBlockMixin, NodeBase):
    """Run Wan Animate 2's official segment denoise/decode loop."""

    label = "Wan Animate 2 Denoise Segments"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "denoise"
    action_key = "workflow_wan_animate_denoise"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "WanAnimate2ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "default", "hidden": True},
        "block_path": {"type": "string", "default": "denoise", "hidden": True},
        "num_inference_steps": {
            "label": "Steps",
            "display": "slider",
            "type": "int",
            "default": 40,
            "min": 1,
            "max": 100,
        },
        "guidance_scale": {
            "label": "Guidance Scale",
            "display": "slider",
            "type": "float",
            "default": 3.0,
            "min": 1.0,
            "max": 20.0,
            "step": 0.1,
        },
        "seed": {
            "label": "Seed",
            "display": "random",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 4294967295,
        },
        "state_out": {"label": "Workflow State", "display": "output", "type": "modular_workflow_state"},
    }

    def execute(self, **kwargs):
        pipeline_class, workflow_id, token, pipeline, state = _wan_animate_state(self, kwargs)
        default_steps = 10 if pipeline_class == _WAN_ANIMATE_DISTILLED_PIPELINE else 40
        steps = _bounded_integer(
            kwargs.get("num_inference_steps", default_steps),
            label="Wan Animate 2 inference steps",
            minimum=1,
            maximum=100,
        )
        guidance_scale = float(kwargs.get("guidance_scale", 3.0))
        if not math.isfinite(guidance_scale) or not 1.0 <= guidance_scale <= 20.0:
            raise ValueError("Wan Animate 2 guidance scale must be finite and from 1 through 20.")
        if pipeline_class == _WAN_ANIMATE_DISTILLED_PIPELINE and guidance_scale != 1.0:
            raise ValueError("Wan Animate 2 Distilled uses the official fixed guidance scale of 1.")
        if getattr(pipeline, "guider", None) is None or not callable(getattr(pipeline.guider, "new", None)):
            raise ValueError("The official Wan Animate 2 denoise block is missing its reviewed guider component.")
        pipeline.update_components(guider=pipeline.guider.new(guidance_scale=guidance_scale))
        state = pipeline(
            state=state,
            num_inference_steps=steps,
            generator=modular_generator_from_seed(kwargs.get("seed", 0), pipeline),
        )
        return _wan_animate_state_result(
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            token=token,
            stage=self.stage,
            state=state,
        )


class WorkflowWanAnimateDecode(_OfficialWorkflowBlockMixin, NodeBase):
    """Assemble Wan Animate 2's decoded segments into one PIL-frame video."""

    label = "Wan Animate 2 Assemble Video"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    stage = "decode"
    action_key = "workflow_wan_animate_decoder"
    params = {
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
            "required": True,
        },
        "state_in": {
            "label": "Workflow State",
            "display": "input",
            "type": "modular_workflow_state",
            "required": True,
        },
        "pipeline_class": {"type": "string", "default": "WanAnimate2ModularPipeline", "hidden": True},
        "workflow_id": {"type": "string", "default": "default", "hidden": True},
        "block_path": {"type": "string", "default": "decode", "hidden": True},
        "video": {"label": "Video", "display": "output", "type": "video"},
    }

    def execute(self, **kwargs):
        _pipeline_class, _workflow_id, _token, pipeline, state = _wan_animate_state(self, kwargs)
        state = pipeline(state=state, output_type="pil")
        videos = state.get("videos")
        if (
            not isinstance(videos, list)
            or len(videos) != 1
            or not isinstance(videos[0], list)
            or not videos[0]
            or not all(isinstance(frame, PILImage.Image) for frame in videos[0])
        ):
            raise ValueError("The official Wan Animate 2 decoder did not return one PIL-frame video.")
        return {"video": videos[0]}
