"""Executable adapters for exact, reviewed Modular Diffusers block placements.

The adapter is intentionally generic only across entries already present in
MoDiff's immutable reviewed node library. It never imports a caller-provided
class or trusts a serialized Python object. Ordinary expanded Block nodes carry
the exact reviewed placement identity and pass a sealed process-local
``PipelineState`` between steps.

Loop members are executable composition inputs rather than one-shot tensor
steps: the reviewed loop owner validates their exact order and executes them on
every upstream timestep, matching ``LoopSequentialPipelineBlocks`` semantics.
"""

from __future__ import annotations

import json
import math
import time
import weakref
from collections.abc import Mapping
from copy import deepcopy

import torch
from diffusers.modular_pipelines import PipelineState

from modiff.NodeBase import NodeBase
from modiff.huggingface_node_library import reviewed_huggingface_node_library
from modiff.modular_block_contracts import reviewed_modular_block_snapshot
from modiff.modular_conditional_contracts import reviewed_modular_conditional_snapshot

from . import components
from .modular_utils import pipeline_class_from_model_type
from .route_state import require_component_binding
from .utils import collect_model_ids


_ISSUER = object()
_ISSUED_STATES: weakref.WeakSet = weakref.WeakSet()
_MAX_LOOP_MEMBERS = 64
_INTEGER_INPUTS = frozenset({"max_sequence_length", "num_images_per_prompt", "height", "width", "num_inference_steps"})
_FLOAT_INPUTS = frozenset({"guidance_scale"})
_JSON_INPUTS = frozenset({"sigmas", "attention_kwargs"})


class _ReviewedNodeProgressBar:
    """Small tqdm-compatible bridge for an exact reviewed loop owner.

    Modular Diffusers loop blocks expose progress through ``progress_bar``
    rather than the standard pipeline ``callback_on_step_end`` hook.  The
    reviewed decomposition executes that loop as an ordinary MoDiff node, so
    bridge its public progress-bar boundary into the normal queue/activity
    telemetry without changing the pinned block tree or inserting an
    unreviewed callback block.
    """

    def __init__(self, node, iterable=None, total=None):
        self._node = node
        self._iterable = iterable
        inferred_total = len(iterable) if total is None and hasattr(iterable, "__len__") else total
        self.total = int(inferred_total) if inferred_total is not None else 0
        self.n = 0
        self._started_at = time.monotonic()

    def __enter__(self):
        self._publish()
        return self

    def __exit__(self, _exception_type, _exception, _traceback):
        return False

    def __iter__(self):
        if self._iterable is None:
            return
        for item in self._iterable:
            yield item
            self.update()

    def close(self):
        return None

    def update(self, amount=1):
        if getattr(self._node, "_interrupt", False):
            raise InterruptedError("Execution interrupted by the user after the current Modular Diffusers step.")
        self.n += int(amount)
        self._publish()

    def _publish(self):
        total = self.total if self.total > 0 else None
        completed = min(self.n, total) if total is not None else self.n
        elapsed = max(0.0, time.monotonic() - self._started_at)
        average = elapsed / completed if completed > 0 else None
        eta = average * max(0, total - completed) if average is not None and total is not None else None
        percent = int(completed / total * 100) if total is not None else -1
        message = f"Denoising {completed}/{total}" if total is not None else "Denoising"
        if completed == 0:
            message += " — first step may include accelerator compilation"
        self._node.progress(
            percent,
            phase="denoising",
            message=message,
            current_step=completed,
            total_steps=total,
            elapsed_seconds=elapsed,
            average_step_seconds=average,
            eta_seconds=eta,
        )


def _field_label(name):
    return " ".join(part.capitalize() for part in str(name).strip("_").split("_") if part) or str(name)


def _reviewed_block_input_params():
    """Publish the pinned Modular block input union through one generic node.

    The client still instantiates only the fields declared by the selected
    immutable block. Publishing the union here makes those per-placement
    fields backend-authored instead of client-invented and lets ordinary
    canvas edges carry media/state values into a reviewed block.
    """

    definitions = reviewed_modular_block_snapshot()["blockDefinitions"]
    fields_by_name = {}
    for block in definitions:
        for field in block.get("inputs", ()):
            name = field.get("name")
            if not isinstance(name, str) or not name or name == "generator":
                continue
            fields_by_name.setdefault(name, []).append(field)

    result = {}
    for name, fields in sorted(fields_by_name.items()):
        type_names = " ".join(str(field.get("type", "opaque")) for field in fields).lower()
        param = {"label": _field_label(name)}
        if "latent" in name or "tensor" in type_names:
            param.update({"display": "input", "type": "latent"})
        elif "pil.image" in type_names or name == "image" or name.endswith("_image") or name.endswith("_images"):
            param.update({"display": "input", "type": "image"})
        elif name in {"video", "driving_video"}:
            param.update({"display": "input", "type": "video"})
        elif name in {"audio", "audios"}:
            param.update({"display": "input", "type": "audio"})
        elif "builtins.bool" in type_names and "opaque" not in type_names:
            param["type"] = "boolean"
        elif "builtins.int" in type_names and "opaque" not in type_names:
            param["type"] = "int"
        elif "builtins.float" in type_names and "opaque" not in type_names:
            param["type"] = "float"
        elif "builtins.str" in type_names and "opaque" not in type_names:
            param["type"] = "string"
            if "prompt" in name or name == "lyrics":
                param["display"] = "textarea"
        else:
            param["type"] = "object"

        defaults = [field.get("default") for field in fields if "default" in field]
        if defaults and all(value == defaults[0] for value in defaults):
            try:
                json.dumps(defaults[0])
            except (TypeError, ValueError, OverflowError, RecursionError):
                pass
            else:
                param["default"] = deepcopy(defaults[0])
        descriptions = [field.get("description") for field in fields if isinstance(field.get("description"), str)]
        if descriptions:
            param["description"] = descriptions[0]
        result[name] = param
    return result


_REVIEWED_BLOCK_INPUT_PARAMS = _reviewed_block_input_params()


class _ReviewedWorkflowState:
    __slots__ = (
        "_token",
        "_pipeline_class",
        "_workflow_id",
        "_execution_scope",
        "_pipeline",
        "_state",
        "_completed_path",
        "_sealed",
        "__weakref__",
    )

    def __init__(self, issuer, *, token, pipeline_class, workflow_id, execution_scope, pipeline, state, completed_path):
        if issuer is not _ISSUER:
            raise TypeError("Reviewed Modular workflow states are backend-issued only.")
        object.__setattr__(self, "_token", token)
        object.__setattr__(self, "_pipeline_class", pipeline_class)
        object.__setattr__(self, "_workflow_id", workflow_id)
        object.__setattr__(self, "_execution_scope", execution_scope)
        object.__setattr__(self, "_pipeline", pipeline)
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_completed_path", tuple(completed_path))
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, _name, _value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Reviewed Modular workflow states are immutable.")
        object.__setattr__(self, _name, _value)

    def __copy__(self):
        return self

    def __deepcopy__(self, _memo):
        return self

    def __reduce_ex__(self, _protocol):
        raise TypeError("Reviewed Modular workflow states cannot be serialized.")


def _issue_state(*, token, pipeline_class, workflow_id, execution_scope, pipeline, state, completed_path):
    issued = _ReviewedWorkflowState(
        _ISSUER,
        token=token,
        pipeline_class=pipeline_class,
        workflow_id=workflow_id,
        execution_scope=execution_scope,
        pipeline=pipeline,
        state=state,
        completed_path=completed_path,
    )
    _ISSUED_STATES.add(issued)
    return issued


def _exact_text(value, *, label):
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty exact string.")
    return value


def _placement_path(value):
    if not isinstance(value, list) or not value or len(value) > 64:
        raise ValueError("Reviewed Modular block placement must be a non-empty bounded array.")
    result = tuple(_exact_text(item, label="Block path segment") for item in value)
    return result


def _runtime_input_value(name, value):
    """Normalize ordinary JSON form values at the exact block boundary.

    Reviewed steps intentionally allow a superset of upstream fields, so the
    generic node cannot use NodeBase's whole-schema casting pass. Apply the
    finite workflow-input contract here instead; this keeps persisted UI
    values byte-stable while giving Diffusers the Python types its blocks
    declare.
    """

    if name in _INTEGER_INPUTS:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, not a boolean.")
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be an integer.") from error
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{name} must be an integer.")
        return number
    if name in _FLOAT_INPUTS:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite number, not a boolean.")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be a finite number.") from error
        if not math.isfinite(number):
            raise ValueError(f"{name} must be a finite number.")
        return number
    if name in _JSON_INPUTS and isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, RecursionError) as error:
            raise ValueError(f"{name} must be valid JSON.") from error
    return value


def _reviewed_definition(pipeline_class, workflow_id):
    matches = [
        definition
        for definition in reviewed_huggingface_node_library()["definitions"]
        if definition.get("provider") == "diffusers"
        and definition.get("pipelineClass") == pipeline_class
        and definition.get("workflowId") == workflow_id
    ]
    if len(matches) != 1:
        raise ValueError(f"No unique reviewed Modular workflow exists for {pipeline_class}/{workflow_id}.")
    return matches[0]


def _reviewed_placement(
    *, pipeline_class, workflow_id, execution_scope, placement_path, block_definition_id, block_class, block_hash
):
    # ``default`` is MoDiff's internal identity for fixed
    # SequentialPipelineBlocks classes that do not expose an upstream
    # ``_workflow_map``. Those classes execute their original nested tree;
    # their selected-workflow catalog uses dotted compatibility keys only for
    # indexing. Validate the real nested path against the immutable unpruned
    # snapshot, exactly as an explicitly unpruned library block does.
    fixed_default_tree = execution_scope == "selected_workflow" and workflow_id == "default"
    if execution_scope == "unpruned_pipeline" or fixed_default_tree:
        snapshot = reviewed_modular_conditional_snapshot()
        pipeline = next(
            (item for item in snapshot["pipelines"] if item.get("pipelineClass") == pipeline_class),
            None,
        )
        if pipeline is None:
            raise ValueError(f"No reviewed unpruned Modular pipeline exists for {pipeline_class}.")
        placement = next(
            (item for item in pipeline["placements"] if tuple(item["path"]) == placement_path),
            None,
        )
        if placement is None or placement.get("blockDefinitionId") != block_definition_id:
            scope_label = "fixed default pipeline" if fixed_default_tree else "unpruned pipeline"
            raise ValueError(f"The Modular block placement is absent from the exact reviewed {scope_label}.")
        block = next(
            (item for item in snapshot["blockDefinitions"] if item.get("id") == block_definition_id),
            None,
        )
        if (
            block is None
            or block.get("className") != block_class
            or block.get("contentHash") != block_hash
        ):
            scope_label = "fixed default" if fixed_default_tree else "unpruned"
            raise ValueError(f"The Modular block identity does not match the reviewed pinned {scope_label} contract.")
        return pipeline, block
    if execution_scope != "selected_workflow":
        raise ValueError("The reviewed Modular execution scope is unsupported.")
    definition = _reviewed_definition(pipeline_class, workflow_id)
    library = reviewed_huggingface_node_library()
    placement = next(
        (item for item in definition["blockPlacements"] if tuple(item["path"]) == placement_path),
        None,
    )
    if placement is None or placement.get("blockDefinitionId") != block_definition_id:
        raise ValueError("The Modular block placement is absent from the exact reviewed workflow.")
    block = next((item for item in library["blockDefinitions"] if item.get("id") == block_definition_id), None)
    if (
        block is None
        or block.get("className") != block_class
        or block.get("contentHash") != block_hash
    ):
        raise ValueError("The Modular block identity does not match the reviewed pinned contract.")
    return definition, block


def _block_at_path(blocks, path):
    current = blocks
    for segment in path:
        sub_blocks = getattr(current, "sub_blocks", None)
        if not isinstance(sub_blocks, Mapping) or segment not in sub_blocks:
            raise ValueError(f"Reviewed Modular block {'.'.join(path)!r} is unavailable at runtime.")
        current = sub_blocks[segment]
    return current


def _component_bundle_token(bundle, *, pipeline_class):
    return require_component_binding(
        bundle,
        label="pipeline component bundle",
        expected_model_type=pipeline_class,
        expected_role="pipeline_components",
    )


def _new_runtime(*, bundle, pipeline_class, workflow_id, execution_scope):
    token = _component_bundle_token(bundle, pipeline_class=pipeline_class)
    pipeline_type = pipeline_class_from_model_type(pipeline_class)
    definition = pipeline_type()
    blocks = definition.blocks
    if execution_scope == "selected_workflow" and workflow_id != "default":
        blocks = blocks.get_workflow(workflow_id)
    pipeline = blocks.init_pipeline(components_manager=components)
    expected = tuple(pipeline.pretrained_component_names)
    model_ids = collect_model_ids(
        {"pipeline_components": bundle},
        target_key_names=("pipeline_components",),
        target_model_names=expected,
    )
    installed = components.get_components_by_ids(ids=model_ids, return_dict_with_names=True) if model_ids else {}
    missing = sorted(set(expected) - set(installed))
    if missing:
        raise ValueError(
            "The connected Models Loader bundle is missing reviewed Modular components: " + ", ".join(missing)
        )
    pipeline.update_components(**installed)
    return token, pipeline, PipelineState()


def _continued_runtime(value, *, bundle, pipeline_class, workflow_id, execution_scope):
    if type(value) is not _ReviewedWorkflowState or value not in _ISSUED_STATES:
        raise ValueError("Connect the backend-issued state from the preceding reviewed Modular block.")
    token = value._token
    if bundle is not None:
        connected_token = _component_bundle_token(bundle, pipeline_class=pipeline_class)
        if token is not connected_token:
            raise ValueError("The connected block state belongs to a different Models Loader execution.")
    if (
        value._pipeline_class != pipeline_class
        or value._workflow_id != workflow_id
        or value._execution_scope != execution_scope
    ):
        raise ValueError("The connected block state belongs to another Modular workflow.")
    return token, value._pipeline, value._state


def _loop_member_descriptor(kwargs, path, block):
    previous = kwargs.get("loop_members_in")
    if previous is None:
        members = []
    elif isinstance(previous, tuple):
        members = [deepcopy(item) for item in previous]
    else:
        raise ValueError("Reviewed loop members must be connected in one exact ordered chain.")
    if len(members) >= _MAX_LOOP_MEMBERS:
        raise ValueError("The reviewed Modular loop member limit was exceeded.")
    members.append(
        {
            "path": list(path),
            "blockDefinitionId": block["id"],
            "blockClass": block["className"],
            "blockContractHash": block["contentHash"],
        }
    )
    return tuple(members)


def _validate_loop_members(block, value, *, parent_path):
    if not isinstance(value, tuple) or not value:
        raise ValueError("The loop owner requires its connected reviewed loop members.")
    expected = [
        {
            "path": [*parent_path, name],
            "blockClass": type(member).__name__,
        }
        for name, member in block.sub_blocks.items()
    ]
    observed = [{"path": item.get("path"), "blockClass": item.get("blockClass")} for item in value]
    if observed != expected:
        raise ValueError(
            "The reviewed loop membership/order changed. Use Fix to restore the exact loop or rebuild the "
            "modified User Node through a supported init_pipeline() composition recipe."
        )


class ReviewedModularWorkflowStep(NodeBase):
    """Run one exact reviewed block, or declare one exact loop member."""

    label = "Reviewed Modular Diffusers Block"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    params = {
        **_REVIEWED_BLOCK_INPUT_PARAMS,
        "pipeline_components": {
            "label": "Pipeline Components",
            "display": "input",
            "type": "diffusers_modular_pipeline_components",
        },
        "state_in": {"label": "Pipeline State", "display": "input", "type": "modular_workflow_state"},
        "loop_members_in": {"label": "Loop Members", "display": "input", "type": "modular_loop_members"},
        "pipeline_class": {"type": "string", "hidden": True},
        "workflow_id": {"type": "string", "hidden": True},
        "execution_scope": {
            "type": "string",
            "hidden": True,
            "default": "selected_workflow",
            "options": ["selected_workflow", "unpruned_pipeline"],
        },
        "placement_path": {"type": "object", "hidden": True},
        "block_definition_id": {"type": "string", "hidden": True},
        "block_class": {"type": "string", "hidden": True},
        "block_contract_hash": {"type": "string", "hidden": True},
        "execution_kind": {
            "type": "string",
            "hidden": True,
            "options": ["step", "loop_owner", "loop_member"],
        },
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "negative_prompt": {"label": "Negative Prompt", "display": "textarea", "type": "text", "default": ""},
        "max_sequence_length": {"label": "Maximum Sequence Length", "type": "int", "default": 1024},
        "num_images_per_prompt": {"label": "Images Per Prompt", "type": "int", "default": 1},
        "latents": {"label": "Latents", "display": "input", "type": "latent"},
        "height": {"label": "Height", "type": "int", "default": 1024},
        "width": {"label": "Width", "type": "int", "default": 1024},
        "seed": {"label": "Seed", "display": "random", "type": "int", "default": 0},
        "num_inference_steps": {"label": "Steps", "type": "int", "default": 50},
        "sigmas": {"label": "Sigmas", "type": "object", "default": None},
        "attention_kwargs": {"label": "Attention kwargs", "type": "object", "default": None},
        "guidance_scale": {"label": "Guidance Scale", "type": "float", "default": 4.0},
        "output_type": {
            "label": "Output Type",
            "type": "string",
            "default": "pil",
            "options": ["pil", "np", "pt"],
        },
        "state_out": {"label": "Pipeline State", "display": "output", "type": "modular_workflow_state"},
        "loop_members": {"label": "Loop Members", "display": "output", "type": "modular_loop_members"},
        "images": {"label": "Images", "display": "output", "type": "image"},
        "videos": {"label": "Videos", "display": "output", "type": "video"},
        "audio": {"label": "Audio", "display": "output", "type": "audio"},
        "audios": {"label": "Audio", "display": "output", "type": "audio"},
        "sampling_rate": {"label": "Sampling Rate", "display": "output", "type": "int"},
        "action": {"label": "Action", "display": "output", "type": "object"},
    }

    def execute(self, **kwargs):
        pipeline_class = _exact_text(kwargs.get("pipeline_class"), label="Pipeline class")
        workflow_id = _exact_text(kwargs.get("workflow_id"), label="Workflow id")
        execution_scope = kwargs.get("execution_scope", "selected_workflow")
        path = _placement_path(kwargs.get("placement_path"))
        block_definition_id = _exact_text(kwargs.get("block_definition_id"), label="Block definition id")
        block_class = _exact_text(kwargs.get("block_class"), label="Block class")
        block_hash = _exact_text(kwargs.get("block_contract_hash"), label="Block contract hash")
        _definition, reviewed_block = _reviewed_placement(
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            execution_scope=execution_scope,
            placement_path=path,
            block_definition_id=block_definition_id,
            block_class=block_class,
            block_hash=block_hash,
        )
        execution_kind = kwargs.get("execution_kind")
        if execution_kind == "loop_member":
            return {
                "state_out": None,
                "loop_members": _loop_member_descriptor(kwargs, path, reviewed_block),
                "images": None,
                "videos": None,
                "audio": None,
                "audios": None,
                "sampling_rate": None,
                "action": None,
            }
        if execution_kind not in {"step", "loop_owner"}:
            raise ValueError("The reviewed Modular block execution kind is unsupported.")

        bundle = kwargs.get("pipeline_components")
        if kwargs.get("state_in") is None:
            token, pipeline, state = _new_runtime(
                bundle=bundle,
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                execution_scope=execution_scope,
            )
        else:
            token, pipeline, state = _continued_runtime(
                kwargs.get("state_in"),
                bundle=bundle,
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                execution_scope=execution_scope,
            )
        runtime_block = _block_at_path(pipeline.blocks, path)
        if type(runtime_block).__name__ != block_class:
            raise ValueError("The runtime Modular block class differs from its pinned reviewed identity.")
        if execution_kind == "loop_owner":
            _validate_loop_members(runtime_block, kwargs.get("loop_members_in"), parent_path=path)

        ignored = {
            "pipeline_components",
            "state_in",
            "loop_members_in",
            "pipeline_class",
            "workflow_id",
            "execution_scope",
            "placement_path",
            "block_definition_id",
            "block_class",
            "block_contract_hash",
            "execution_kind",
        }
        block_input_specs = {item.name: item for item in runtime_block.inputs if item.name}
        block_inputs = set(block_input_specs)
        for name, value in kwargs.items():
            if name in ignored or value is None or name not in block_inputs:
                continue
            if name == "seed":
                continue
            state.set(
                name,
                _runtime_input_value(name, value),
                kwargs_type=block_input_specs[name].kwargs_type,
            )
        if "generator" in block_inputs and kwargs.get("seed") is not None:
            device = getattr(pipeline, "_execution_device", None) or "cpu"
            state.set("generator", torch.Generator(device=device).manual_seed(int(kwargs["seed"])))
        guidance_scale = kwargs.get("guidance_scale")
        if guidance_scale is not None and "guider" in pipeline.component_names:
            guider_spec = pipeline.get_component_spec("guider")
            pipeline.update_components(guider=guider_spec.create(guidance_scale=float(guidance_scale)))

        progress_bar_override = execution_kind == "loop_owner" and callable(
            getattr(runtime_block, "progress_bar", None)
        )
        had_instance_progress_bar = "progress_bar" in vars(runtime_block)
        previous_instance_progress_bar = vars(runtime_block).get("progress_bar")
        if progress_bar_override:
            runtime_block.progress_bar = lambda iterable=None, total=None: _ReviewedNodeProgressBar(
                self,
                iterable=iterable,
                total=total,
            )
        try:
            _pipeline, state = runtime_block(pipeline, state)
        finally:
            if progress_bar_override:
                if had_instance_progress_bar:
                    runtime_block.progress_bar = previous_instance_progress_bar
                else:
                    del runtime_block.progress_bar
        declared_outputs = {item.get("name") for item in reviewed_block.get("outputs", ())}
        published = {
            name: state.get(name) if name in declared_outputs else None
            for name in ("images", "videos", "audio", "audios", "sampling_rate", "action")
        }
        return {
            "state_out": _issue_state(
                token=token,
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                execution_scope=execution_scope,
                pipeline=pipeline,
                state=state,
                completed_path=path,
            ),
            "loop_members": None,
            **published,
        }
