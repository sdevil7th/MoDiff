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
from contextlib import nullcontext
from collections.abc import Mapping
from copy import deepcopy
from types import UnionType
from typing import Union, get_args, get_origin

import torch
from diffusers.modular_pipelines import PipelineState

from modiff.NodeBase import NodeBase
from modiff.huggingface_node_library import reviewed_huggingface_node_library
from modiff.modular_block_contracts import reviewed_modular_block_snapshot
from modiff.modular_conditional_contracts import reviewed_modular_conditional_snapshot
from modiff.modular_composition import (
    build_reviewed_modular_composition_blocks,
    validate_modular_composition_recipe,
)
from modiff.modular_loop_bindings import bind_upstream_loop_inputs
from modiff.modular_requirements import validate_runtime_component_requirements
from modiff.modular_runtime_diagnostics import modular_execution_diagnostics

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


def _prepare_reviewed_block_components(pipeline_class, block_class, pipeline):
    """Onload Helios' VAE before the pinned encoder reads its device.

    The image/video blocks create normalization tensors before ``encode``
    triggers its offload hook. Use that same upstream hook early so those
    tensors follow the VAE, retaining the manager's eviction strategy and hooks.
    This is selected only after the exact reviewed placement is validated.
    """
    if pipeline_class not in {
        "HeliosModularPipeline", "HeliosPyramidModularPipeline", "HeliosPyramidDistilledModularPipeline"
    } or block_class not in {"HeliosImageVaeEncoderStep", "HeliosVideoVaeEncoderStep"}:
        return
    from diffusers.modular_pipelines.components_manager import CustomOffloadHook

    vae = pipeline.vae
    hook = getattr(vae, "_hf_hook", None)
    if isinstance(hook, CustomOffloadHook) and vae.device != hook.execution_device:
        hook.pre_forward(vae)


def _field_label(name):
    return " ".join(part.capitalize() for part in str(name).strip("_").split("_") if part) or str(name)


def _reviewed_block_input_params(field_kind="inputs"):
    """Publish the pinned Modular block input union through one generic node.

    The client still instantiates only the fields declared by the selected
    immutable block. Publishing the union here makes those per-placement
    fields backend-authored instead of client-invented and lets ordinary
    canvas edges carry media/state values into a reviewed block.
    """

    definitions = reviewed_modular_block_snapshot()["blockDefinitions"]
    if field_kind == "outputs":
        definitions = [*definitions, *reviewed_modular_conditional_snapshot()["blockDefinitions"]]
    fields_by_name = {}
    for block in definitions:
        for field in block.get(field_kind, ()):
            name = field.get("name")
            if not isinstance(name, str) or not name or (name == "generator" and field_kind == "inputs"):
                continue
            fields_by_name.setdefault(name, []).append(field)

    result = {}
    for name, fields in sorted(fields_by_name.items()):
        type_names = " ".join(str(field.get("type", "opaque")) for field in fields).lower()
        param = {"label": _field_label(name)}
        if name == "generator":
            param.update({"display": "input", "type": "generator"})
        elif "latent" in name or "tensor" in type_names:
            param.update({"display": "input", "type": "latent"})
        elif name in {"video", "driving_video"}:
            # Upstream video processors may annotate a sequence of PIL frames.
            # Preserve the video socket that supplies that decoded sequence.
            param.update({"display": "input", "type": "video"})
        elif "pil.image" in type_names or name == "image" or name.endswith("_image") or name.endswith("_images"):
            param.update({"display": "input", "type": "image"})
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
_REVIEWED_BLOCK_INPUT_ALIASES = {
    f"state_input__{name}": {
        key: value for key, value in {**param, "display": "input"}.items()
        if key not in {"default", "value", "required", "options"}
    }
    for name, param in _REVIEWED_BLOCK_INPUT_PARAMS.items()
}
_REVIEWED_BLOCK_OUTPUT_PARAMS = {
    f"state_output__{name}": {
        key: value for key, value in {**param, "display": "output"}.items()
        if key not in {"default", "value", "required", "options"}
    }
    for name, param in _reviewed_block_input_params("outputs").items()
}
_REVIEWED_LOOP_PORT_PARAMS = {
    **{f"iteration_input__{name}": {
        **{key: value for key, value in param.items() if key not in {"default", "value", "required", "options"}},
        "display": "input", "type": list(dict.fromkeys([*(param["type"] if isinstance(param["type"], list) else [param["type"]]), "modular_loop_value"])),
        "label": f"{param.get('label', name)} (each iteration)",
    } for name, param in _REVIEWED_BLOCK_INPUT_PARAMS.items()},
    **{f"{prefix}{name}": {"display": "output", "type": "modular_loop_value", "label": f"{name} ({label})"}
       for name in _reviewed_block_input_params("outputs")
       for prefix, label in (("iteration_output__", "this iteration"), ("iteration_previous__", "previous iteration"))},
}


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


def _numeric_form_contract(type_hint):
    if type_hint in (int, float, bool, type(None)):
        return True
    origin = get_origin(type_hint)
    args = get_args(type_hint)
    return bool(args) and origin in (list, Union, UnionType) and all(_numeric_form_contract(arg) for arg in args)


def _numeric_form_value(name, value, type_hint):
    """Convert only an exact numeric/bool declaration, never opaque/text unions."""
    origin = get_origin(type_hint)
    if origin in (Union, UnionType):
        for candidate in get_args(type_hint):
            try:
                return _numeric_form_value(name, value, candidate)
            except ValueError:
                pass
        raise ValueError(f"{name} must match its declared numeric or boolean input type.")
    if origin is list:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, RecursionError) as error:
                raise ValueError(f"{name} must be a valid numeric list.") from error
        if not isinstance(value, list):
            raise ValueError(f"{name} must be a numeric list.")
        return [_numeric_form_value(name, item, get_args(type_hint)[0]) for item in value]
    if type_hint is type(None):
        if value is None:
            return None
        raise ValueError(f"{name} must be null.")
    if type_hint is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        raise ValueError(f"{name} must be a boolean.")
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not a boolean.")
    try:
        number = type_hint(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite {type_hint.__name__}.") from error
    if not math.isfinite(number) or (type_hint is int and isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{name} must be a finite {type_hint.__name__}.")
    return number


def _runtime_input_value(name, value, *, type_hint=None):
    """Normalize ordinary JSON form values at the exact block boundary.

    Reviewed steps intentionally allow a superset of upstream fields, so the
    generic node cannot use NodeBase's whole-schema casting pass. Apply the
    exact upstream scalar/list type here instead; this keeps persisted UI
    values byte-stable while giving Diffusers the Python types its blocks
    declare.
    """

    if type_hint is not None and _numeric_form_contract(type_hint):
        return _numeric_form_value(name, value, type_hint)
    if type_hint is not None and name not in _JSON_INPUTS:
        # A text/opaque alternative is meaningful. Do not override it with a
        # historical field-name guess used for descriptors without type hints.
        return value
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


def _reviewed_runtime_inputs(pipeline_class, values, input_specs):
    """Translate the reviewed Helios stage controls without rewriting the graph.

    Registered V2 workflows retain the legacy scalar controls alongside the
    upstream list default. Each supplied scalar owns its corresponding list
    entry, just as in WorkflowVideoDenoise; a native-list-only workflow keeps
    its original values and stage count.
    """

    target = "pyramid_num_inference_steps_list"
    if pipeline_class not in {
        "HeliosPyramidModularPipeline",
        "HeliosPyramidDistilledModularPipeline",
    } or target not in input_specs:
        return values
    aliases = tuple(f"pyramid_stage_{stage}_steps" for stage in (1, 2, 3))
    if not any(values.get(name) is not None for name in aliases):
        return values
    native = values.get(target)
    if native is None:
        native = input_specs[target].default
    if not isinstance(native, (list, tuple)) or len(native) != 3:
        raise ValueError("Helios scalar pyramid stage controls require a native list with three entries.")
    stages = list(native)
    for index, name in enumerate(aliases):
        value = values.get(name)
        if value is None:
            continue
        try:
            number = _runtime_input_value("num_inference_steps", value)
        except ValueError as error:
            raise ValueError(f"{name} must be an integer from 1 through 50.") from error
        if not 1 <= number <= 50:
            raise ValueError(f"{name} must be an integer from 1 through 50.")
        stages[index] = number
    return {**values, target: stages}


def _reviewed_resolved_dimensions(pipeline_class, block_class, state, values):
    """Wan's image step owns resolved geometry after interpreting target area.

    Repeated creator controls on downstream nodes still contain target-area
    dimensions. Native video/denoise steps must retain the resolved dimensions
    carried by the authenticated upstream state instead of resetting them.
    """
    if pipeline_class not in {"WanAnimate2ModularPipeline", "WanAnimate2DistilledModularPipeline"}:
        return values
    if block_class not in {
        "WanAnimate2ProcessVideosInputStep", "WanAnimate2DenoiseStep", "WanAnimate2DistilledDenoiseStep",
    } or state.get("image_pixels") is None:
        return values
    return {**values, "height": state.get("height"), "width": state.get("width")}


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
    *, pipeline_class, workflow_id, execution_scope, placement_path, block_definition_id, block_class, block_hash,
    composition=None,
):
    if composition is not None:
        if (
            composition["pipelineClass"] != pipeline_class
            or composition["workflowId"] != workflow_id
            or execution_scope != "unpruned_pipeline"
        ):
            raise ValueError("The edited Modular composition belongs to another workflow or execution scope.")
        placement = next(
            (item for item in composition["composedPlacements"] if tuple(item["path"]) == placement_path), None,
        )
        block = next(
            (item for item in reviewed_modular_conditional_snapshot()["blockDefinitions"]
             if item.get("id") == block_definition_id), None,
        )
        if (
            placement is None or placement["blockDefinitionId"] != block_definition_id
            or block is None or block.get("className") != block_class or block.get("contentHash") != block_hash
        ):
            raise ValueError("The Modular block identity/path does not match the validated edited composition.")
        return composition, block
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


def _new_runtime(*, bundle, pipeline_class, workflow_id, execution_scope, composition_recipe=None):
    token = _component_bundle_token(bundle, pipeline_class=pipeline_class)
    composition_hash = None
    if composition_recipe is not None:
        recipe, blocks = build_reviewed_modular_composition_blocks(composition_recipe)
        composition_hash = recipe["recipeHash"]
    else:
        pipeline_type = pipeline_class_from_model_type(pipeline_class)
        definition = pipeline_type()
        blocks = definition.blocks
        if execution_scope == "selected_workflow" and workflow_id != "default":
            blocks = blocks.get_workflow(workflow_id)
    pipeline = blocks.init_pipeline(components_manager=components)
    pipeline._modiff_composition_hash = composition_hash
    expected = tuple(pipeline.pretrained_component_names)
    # An edited graph explicitly invokes its reviewed steps. Do not re-run the
    # original conditional selectors to guess the graph's component demand:
    # their selected child can legitimately have moved to another container.
    # Install the connected bundle's applicable components; each executed step
    # still validates its exact required components/types before invocation.
    # Inactive branches must not force extra models into a text-only workflow.
    model_ids = collect_model_ids(
        {"pipeline_components": bundle},
        target_key_names=("pipeline_components",),
        target_model_names=expected,
    )
    installed = components.get_components_by_ids(ids=model_ids, return_dict_with_names=True) if model_ids else {}
    missing = sorted(set(expected) - set(installed)) if composition_recipe is None else []
    if missing:
        raise ValueError(
            "The connected Models Loader bundle is missing reviewed Modular components: " + ", ".join(missing)
        )
    pipeline.update_components(**installed)
    if pipeline_class == "HeliosPyramidModularPipeline":
        # Selected upstream trees retain an aggregate CFG spec. The expanded
        # executor creates its own pipeline and transfers only pretrained
        # components, so the loader's corrected from-config guider is absent.
        # Use the actual pinned inner denoiser spec before text encoding and
        # retain it for later guidance-scale component recreation.
        from diffusers.modular_pipelines.helios.denoise import HeliosPyramidChunkDenoiseInner

        guider_spec = next(
            spec for spec in HeliosPyramidChunkDenoiseInner().expected_components if spec.name == "guider"
        )
        pipeline.update_components(guider=guider_spec.create())
    return token, pipeline, PipelineState()


def _continued_runtime(value, *, bundle, pipeline_class, workflow_id, execution_scope, composition_hash=None):
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
    if getattr(value._pipeline, "_modiff_composition_hash", None) != composition_hash:
        raise ValueError("The connected Pipeline State belongs to a different edited Modular composition. Re-run its upstream nodes.")
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
    bindings = kwargs.get("iteration_bindings")
    if bindings is not None:
        declared = {field["name"] for field in block.get("inputs", ())}
        if not isinstance(bindings, dict) or len(bindings) > 128 or set(bindings) - declared:
            raise ValueError(f"Loop member {'/'.join(path)}: iteration bindings must target its exact declared inputs.")
    bindings = dict(bindings or {})
    declared = {field["name"] for field in block.get("inputs", ())}
    for key, value in kwargs.items():
        if not key.startswith("iteration_input__") or value is None:
            continue
        field = key.removeprefix("iteration_input__")
        if field not in declared:
            raise ValueError(f"Loop member {'/'.join(path)}: input {field!r} is not declared.")
        if field in bindings:
            raise ValueError(f"Loop member {'/'.join(path)}, input {field}: disconnect one of the competing iteration drivers.")
        bindings[field] = {"kind": "constant", "value": value}
    members.append(
        {
            "path": list(path),
            "blockDefinitionId": block["id"],
            "blockClass": block["className"],
            "blockContractHash": block["contentHash"],
            **({"iterationBindings": deepcopy(bindings)} if bindings else {}),
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


def _published_outputs(state, declared_outputs):
    published = {
        name: state.get(name) if name in declared_outputs else None
        for name in ("images", "videos", "audio", "audios", "sound", "sampling_rate", "action")
    }
    # Keep the official waveform untouched inside PipelineState. The ordinary
    # canvas audio socket carries MoDiff's sample-rate-bearing audio object,
    # matching the existing workflow decoder/export boundary (no default Hz).
    if published["sound"] is not None:
        sample_rate = state.get("sampling_rate")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("The reviewed sound output requires its actual positive integer sampling_rate.")
        from modules.DiffusersAudio.main import output_to_audio_objects

        audio = output_to_audio_objects(published["sound"], sample_rate=sample_rate)
        if len(audio) != 1:
            raise ValueError("The reviewed sound socket requires one waveform; batch audio cannot be silently discarded.")
        published["sound"] = audio[0]
    # Inputs and outputs may have the same upstream name. Separate ordinary
    # output handles avoid overwriting the editable input. Only the exact
    # reviewed block's declared outputs are published, never arbitrary state.
    published.update({
        f"state_output__{name}": state.get(name)
        for name in declared_outputs
        if f"state_output__{name}" in _REVIEWED_BLOCK_OUTPUT_PARAMS
    })
    return published


class ReviewedModularWorkflowStep(NodeBase):
    """Run one exact reviewed block, or declare one exact loop member."""

    label = "Reviewed Modular Diffusers Block"
    category = "Modular Diffusers"
    resizable = True
    skipParamsCheck = True
    params = {
        **_REVIEWED_BLOCK_INPUT_PARAMS,
        **_REVIEWED_BLOCK_INPUT_ALIASES,
        **_REVIEWED_BLOCK_OUTPUT_PARAMS,
        **_REVIEWED_LOOP_PORT_PARAMS,
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
        "composition_recipe": {"type": "object", "hidden": True},
        "iteration_bindings": {"type": "object", "hidden": True},
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
        "generator": {"label": "Generator", "display": "input", "type": "generator"},
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
        "sound": {"label": "Sound", "display": "output", "type": "audio"},
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
        composition_recipe = kwargs.get("composition_recipe")
        composition = (
            validate_modular_composition_recipe(composition_recipe) if composition_recipe is not None else None
        )
        _definition, reviewed_block = _reviewed_placement(
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            execution_scope=execution_scope,
            placement_path=path,
            block_definition_id=block_definition_id,
            block_class=block_class,
            block_hash=block_hash,
            composition=composition,
        )
        execution_kind = kwargs.get("execution_kind")
        if execution_kind == "loop_member":
            self.record_generation_inputs({"iteration_bindings": kwargs["iteration_bindings"]} if kwargs.get("iteration_bindings") else {})
            return {
                "state_out": None,
                "loop_members": _loop_member_descriptor(kwargs, path, reviewed_block),
                "images": None,
                "videos": None,
                "audio": None,
                "audios": None,
                "sound": None,
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
                **({"composition_recipe": composition_recipe} if composition is not None else {}),
            )
        else:
            token, pipeline, state = _continued_runtime(
                kwargs.get("state_in"),
                bundle=bundle,
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                execution_scope=execution_scope,
                **({"composition_hash": composition["recipeHash"]} if composition is not None else {}),
            )
        runtime_block = _block_at_path(pipeline.blocks, path)
        if type(runtime_block).__name__ != block_class:
            raise ValueError("The runtime Modular block class differs from its pinned reviewed identity.")
        validate_runtime_component_requirements(runtime_block, pipeline, path=path)
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
            "composition_recipe",
            "iteration_bindings",
        }
        block_input_specs = {item.name: item for item in runtime_block.inputs if item.name}
        block_inputs = set(block_input_specs)
        consumed_inputs = {}
        bound_inputs = dict(kwargs)
        for alias in _REVIEWED_BLOCK_INPUT_ALIASES:
            value = kwargs.get(alias)
            if value is None:
                continue
            name = alias.removeprefix("state_input__")
            if name not in block_inputs:
                raise ValueError(f"{block_class} does not declare an input named {name!r}.")
            if kwargs.get(name) is not None:
                raise ValueError(f"{block_class}.{name} has competing direct and aliased input values. Disconnect one driver.")
            bound_inputs[name] = value
        runtime_inputs = _reviewed_runtime_inputs(pipeline_class, bound_inputs, block_input_specs)
        runtime_inputs = _reviewed_resolved_dimensions(pipeline_class, block_class, state, runtime_inputs)
        for name, value in runtime_inputs.items():
            if name in ignored or value is None or name not in block_inputs:
                continue
            if name == "seed":
                continue
            value = _runtime_input_value(name, value, type_hint=getattr(block_input_specs[name], "type_hint", None))
            state.set(name, value, kwargs_type=block_input_specs[name].kwargs_type)
            consumed_inputs[name] = value
        # Legacy misplaced seeds remain inert until the user applies the explicit
        # graph repair. Initializing on an unrelated preparation step would change
        # saved workflows and can reset an already-advanced shared Generator.
        if "generator" in block_inputs and kwargs.get("generator") is None and kwargs.get("seed") is not None:
            device = getattr(pipeline, "_execution_device", None) or "cpu"
            seed = int(kwargs["seed"])
            state.set("generator", torch.Generator(device=device).manual_seed(seed))
            consumed_inputs["seed"] = seed
        guidance_scale = kwargs.get("guidance_scale")
        if guidance_scale is not None and "guider" in pipeline.component_names:
            guider_spec = pipeline.get_component_spec("guider")
            guidance_scale = float(guidance_scale)
            pipeline.update_components(guider=guider_spec.create(guidance_scale=guidance_scale))
            consumed_inputs["guidance_scale"] = guidance_scale

        # A reviewed adapter accepts a union of fields but a particular block
        # consumes only its exact inputs. Capture those normalized values, not
        # unused union defaults or the pre-conversion persisted UI strings.
        self.record_generation_inputs(consumed_inputs)

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
            _prepare_reviewed_block_components(pipeline_class, block_class, pipeline)
            bindings = (
                bind_upstream_loop_inputs(runtime_block, kwargs["loop_members_in"], parent_path=path, coerce_input=_runtime_input_value)
                if execution_kind == "loop_owner" else nullcontext()
            )
            with bindings, modular_execution_diagnostics(runtime_block, state, path=path):
                _pipeline, state = runtime_block(pipeline, state)
        finally:
            if progress_bar_override:
                if had_instance_progress_bar:
                    runtime_block.progress_bar = previous_instance_progress_bar
                else:
                    del runtime_block.progress_bar
        declared_outputs = {item.get("name") for item in reviewed_block.get("outputs", ())}
        published = _published_outputs(state, declared_outputs)
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
