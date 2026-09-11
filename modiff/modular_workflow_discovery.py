"""Normalized, bounded contracts derived from pinned Modular Diffusers blocks.

Runtime consumers read a reviewed generated snapshot. Regeneration and tests
pass already-instantiated no-weight upstream blocks into the builder, keeping
ordinary registry discovery free of model-stack imports and network access.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, get_args, get_origin

from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION


MODULAR_WORKFLOW_CONTRACT_SCHEMA_VERSION = 1
MODULAR_WORKFLOW_SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / "modular-workflow-contracts.json"
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,255}$")
_MAX_CONTRACTS = 64
_MAX_WORKFLOWS = 128
_MAX_FIELDS = 512
_MAX_STEPS = 1024


class ModularWorkflowContractError(ValueError):
    """A normalized upstream workflow contract is malformed or ambiguous."""


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ModularWorkflowContractError(f"{label} {value!r} must be a bounded identifier.")
    return value


def _description(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:1024]


def _block_description(block: Any) -> str:
    try:
        value = block.description
    except (AttributeError, NotImplementedError):
        value = ""
    return _description(value)


def _json_default(value: Any) -> Any:
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, (list, tuple)) and len(value) <= 32:
        normalized = [_json_default(item) for item in value]
        if all(item is not None or source is None for item, source in zip(normalized, value)):
            return normalized
    return None


def _type_name(type_hint: Any) -> str:
    if type_hint is None:
        return "opaque"
    origin = get_origin(type_hint)
    if origin is not None:
        args = sorted({_type_name(item) for item in get_args(type_hint)})
        origin_name = getattr(origin, "__name__", str(origin).rsplit(".", 1)[-1])
        return f"{origin_name}[{','.join(args)}]"[:256]
    module = getattr(type_hint, "__module__", "")
    name = getattr(type_hint, "__name__", None)
    if isinstance(name, str):
        return f"{module}.{name}".strip(".")[:256]
    return str(type_hint)[:256]


def _field_contract(field: Any, *, required_names: set[str]) -> dict[str, Any]:
    name = _name(getattr(field, "name", None), "Workflow field name")
    return {
        "name": name,
        "type": _type_name(getattr(field, "type_hint", None)),
        "required": name in required_names or getattr(field, "required", False) is True,
        "default": _json_default(getattr(field, "default", None)),
        "description": _description(getattr(field, "description", "")),
    }


def _block_kind(block: Any) -> str:
    names = {base.__name__ for base in type(block).__mro__}
    if "LoopSequentialPipelineBlocks" in names:
        return "loop"
    if "AutoPipelineBlocks" in names or getattr(block, "_workflow_map", None):
        return "auto"
    if "SequentialPipelineBlocks" in names:
        return "sequential"
    return "block"


def _block_steps(blocks: Any) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []

    def visit(block: Any, path: str) -> None:
        if len(steps) >= _MAX_STEPS:
            raise ModularWorkflowContractError("Workflow exceeds the normalized block-step limit.")
        steps.append(
            {
                "path": _name(path, "Workflow block path"),
                "className": _name(type(block).__name__, "Workflow block class"),
                "kind": _block_kind(block),
                "description": _block_description(block),
            }
        )
        nested = getattr(block, "sub_blocks", None)
        if not isinstance(nested, Mapping):
            return
        for child_name, child in nested.items():
            child_path = f"{path}.{child_name}" if path else str(child_name)
            visit(child, child_path)

    nested = getattr(blocks, "sub_blocks", None)
    if not isinstance(nested, Mapping):
        raise ModularWorkflowContractError("Upstream workflow blocks do not expose a sub-block mapping.")
    for block_name, block in nested.items():
        visit(block, str(block_name))
    return steps


def _component_contracts(pipeline: Any) -> list[dict[str, Any]]:
    components = []
    for component in getattr(pipeline.blocks, "expected_components", ()):
        name = _name(getattr(component, "name", None), "Component name")
        type_hint = getattr(component, "type_hint", None)
        components.append(
            {
                "name": name,
                "type": _type_name(type_hint),
                "creationMethod": str(getattr(component, "default_creation_method", ""))[:64],
                "reuseKey": [name, "loadId", "dtype", "quantization", "device", "offloadMode"],
            }
        )
    if len(components) > _MAX_FIELDS or len({item["name"] for item in components}) != len(components):
        raise ModularWorkflowContractError("Component contract is oversized or contains duplicate names.")
    return components


def _workflow_requirement_sets(value: Any, workflow_id: str) -> list[set[str]]:
    """Normalize one upstream workflow predicate, including OR alternatives."""

    if isinstance(value, Mapping):
        candidates = (value,)
    elif isinstance(value, tuple) and value and all(isinstance(item, Mapping) for item in value):
        candidates = value
    else:
        raise ModularWorkflowContractError(
            f"Workflow {workflow_id!r} has an unsupported upstream requirement map."
        )
    return [
        {str(name) for name, required in candidate.items() if required is True}
        for candidate in candidates
    ]


def build_modular_workflow_contract(
    pipeline: Any,
    *,
    aliases: Mapping[str, str] | None = None,
    ui_defaults: Mapping[str, Any] | None = None,
    pipeline_state_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Derive one normalized contract from no-weight upstream APIs."""

    pipeline_class = _name(type(pipeline).__name__, "Pipeline class")
    blocks = pipeline.blocks
    blocks_class = _name(type(blocks).__name__, "Blocks class")
    aliases = dict(aliases or {})
    ui_defaults = dict(ui_defaults or {})
    raw_map = getattr(blocks, "_workflow_map", None)
    workflow_map = raw_map if isinstance(raw_map, Mapping) and raw_map else None
    if workflow_map is not None:
        available = list(blocks.available_workflows)
        if set(available) != set(workflow_map):
            raise ModularWorkflowContractError("available_workflows disagrees with the upstream workflow map.")
    else:
        try:
            available = list(blocks.available_workflows)
        except NotImplementedError:
            available = []
        if available:
            raise ModularWorkflowContractError("A non-Auto block unexpectedly advertises named workflows.")
        available = ["default"]
    if len(available) > _MAX_WORKFLOWS or len(set(available)) != len(available):
        raise ModularWorkflowContractError("Upstream workflows are oversized or duplicated.")

    workflows = []
    for workflow_id in available:
        workflow_id = _name(workflow_id, "Workflow id")
        selected_workflow = blocks.get_workflow(workflow_id) if workflow_map is not None else blocks
        workflow = selected_workflow.get_execution_blocks()
        initialized = workflow.init_pipeline()
        execution_pipeline_class = _name(type(initialized).__name__, "Execution pipeline class")
        input_fields = [field for field in workflow.inputs if isinstance(getattr(field, "name", None), str)]
        input_names = {field.name for field in input_fields}
        workflow_requirement_sets = (
            _workflow_requirement_sets(workflow_map[workflow_id], workflow_id)
            if workflow_map is not None
            else [set()]
        )
        # Auto workflow maps may also contain selector-only predicates (for
        # example Cosmos ``enable_sound``) that are not execution inputs. Only
        # publish requirements the selected execution blocks can receive.
        intrinsic_required = {
            str(field.name) for field in input_fields if getattr(field, "required", False) is True
        }
        filtered_requirement_sets = [
            (requirements & input_names) | intrinsic_required for requirements in workflow_requirement_sets
        ]
        required_inputs = set.intersection(*filtered_requirement_sets)
        distinct_requirement_sets: list[set[str]] = []
        for requirements in filtered_requirement_sets:
            if requirements not in distinct_requirement_sets:
                distinct_requirement_sets.append(requirements)
        inputs = [_field_contract(field, required_names=required_inputs) for field in input_fields]
        outputs = [
            _field_contract(field, required_names=set())
            for field in workflow.outputs
            if isinstance(getattr(field, "name", None), str)
        ]
        if len(inputs) > _MAX_FIELDS or len(outputs) > _MAX_FIELDS:
            raise ModularWorkflowContractError(f"Workflow {workflow_id!r} exceeds the field limit.")
        state_keys: list[str] = []
        if pipeline_state_factory is not None:
            required_values = {name: object() for name in required_inputs}
            try:
                pipeline_state = pipeline_state_factory(values=required_values)
            except TypeError:
                pipeline_state = pipeline_state_factory()
                setter = getattr(pipeline_state, "set", None)
                if not callable(setter):
                    raise ModularWorkflowContractError("Pipeline state factory cannot publish required inputs.")
                for name, value in required_values.items():
                    setter(name, value)
            state = workflow.get_block_state(pipeline_state)
            values = getattr(state, "__dict__", None)
            if not isinstance(values, dict):
                raise ModularWorkflowContractError(f"Workflow {workflow_id!r} returned an invalid block state.")
            state_keys = [_name(name, "State key") for name in values]
        task_id = _name(aliases.get(workflow_id, workflow_id), "Task id")
        workflow_contract = {
            "id": workflow_id,
            "taskId": task_id,
            "label": task_id.replace("_", " ").replace("-", " ").title(),
            "executionPipelineClass": execution_pipeline_class,
            "kind": _block_kind(workflow),
            "requiredInputs": sorted(required_inputs),
            "inputs": inputs,
            "outputs": outputs,
            "stateKeys": state_keys,
            "steps": _block_steps(workflow),
        }
        if len(distinct_requirement_sets) > 1:
            workflow_contract["requiredInputAlternatives"] = sorted(
                (sorted(requirements) for requirements in distinct_requirement_sets),
                key=lambda names: tuple(names),
            )
        workflows.append(workflow_contract)
    if len({workflow["taskId"] for workflow in workflows}) != len(workflows):
        raise ModularWorkflowContractError("Reviewed workflow aliases must be unique.")
    unknown_aliases = set(aliases) - {workflow["id"] for workflow in workflows}
    if unknown_aliases:
        raise ModularWorkflowContractError(f"Reviewed aliases target unknown workflows: {sorted(unknown_aliases)}")

    defaults = {}
    all_input_names = {field["name"] for workflow in workflows for field in workflow["inputs"]}
    for name, value in ui_defaults.items():
        if name not in all_input_names:
            raise ModularWorkflowContractError(f"UI default targets unknown input {name!r}.")
        normalized = _json_default(value)
        if normalized is None and value is not None:
            raise ModularWorkflowContractError(f"UI default for {name!r} is not JSON-safe.")
        defaults[name] = normalized
    return validate_modular_workflow_contract(
        {
            "schemaVersion": MODULAR_WORKFLOW_CONTRACT_SCHEMA_VERSION,
            "pipelineClass": pipeline_class,
            "blocksClass": blocks_class,
            "kind": "auto" if workflow_map is not None else _block_kind(blocks),
            "description": _block_description(blocks),
            "uiDefaults": defaults,
            "components": _component_contracts(pipeline),
            "workflows": workflows,
        }
    )


def validate_modular_workflow_contract(value: Any) -> dict[str, Any]:
    """Validate and JSON-round-trip one normalized schema-v1 contract."""

    try:
        normalized = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as error:
        raise ModularWorkflowContractError(f"Workflow contract is not finite JSON: {error}") from error
    if not isinstance(normalized, dict) or normalized.get("schemaVersion") != 1:
        raise ModularWorkflowContractError("Workflow contract requires schemaVersion 1.")
    if set(normalized) != {
        "schemaVersion",
        "pipelineClass",
        "blocksClass",
        "kind",
        "description",
        "uiDefaults",
        "components",
        "workflows",
    }:
        raise ModularWorkflowContractError("Workflow contract has missing or unknown top-level fields.")
    _name(normalized.get("pipelineClass"), "Pipeline class")
    _name(normalized.get("blocksClass"), "Blocks class")
    if normalized.get("kind") not in {"auto", "sequential", "loop", "block"}:
        raise ModularWorkflowContractError("Workflow contract has an invalid kind.")
    if not isinstance(normalized.get("description"), str) or len(normalized["description"]) > 1024:
        raise ModularWorkflowContractError("Workflow description is invalid.")
    if not isinstance(normalized.get("uiDefaults"), dict) or len(normalized["uiDefaults"]) > _MAX_FIELDS:
        raise ModularWorkflowContractError("Workflow UI defaults are invalid.")
    components = normalized.get("components")
    workflows = normalized.get("workflows")
    if not isinstance(components, list) or len(components) > _MAX_FIELDS:
        raise ModularWorkflowContractError("Workflow components are invalid.")
    if not isinstance(workflows, list) or not workflows or len(workflows) > _MAX_WORKFLOWS:
        raise ModularWorkflowContractError("Workflow entries are invalid.")
    component_names = set()
    for component in components:
        if not isinstance(component, dict) or set(component) != {
            "name",
            "type",
            "creationMethod",
            "reuseKey",
        }:
            raise ModularWorkflowContractError("Component entry is malformed.")
        name = _name(component.get("name"), "Component name")
        if name in component_names:
            raise ModularWorkflowContractError("Component names must be unique.")
        component_names.add(name)
        if not all(isinstance(component.get(key), str) for key in ("type", "creationMethod")):
            raise ModularWorkflowContractError("Component type metadata is malformed.")
        if component.get("reuseKey") != [name, "loadId", "dtype", "quantization", "device", "offloadMode"]:
            raise ModularWorkflowContractError("Component reuse metadata is malformed.")
    workflow_ids = set()
    task_ids = set()
    all_input_names = set()
    for workflow in workflows:
        workflow_keys = {
            "id",
            "taskId",
            "label",
            "executionPipelineClass",
            "kind",
            "requiredInputs",
            "inputs",
            "outputs",
            "stateKeys",
            "steps",
        }
        if not isinstance(workflow, dict) or set(workflow) not in {
            frozenset(workflow_keys),
            frozenset(workflow_keys | {"requiredInputAlternatives"}),
        }:
            raise ModularWorkflowContractError("Workflow entry is malformed.")
        workflow_id = _name(workflow.get("id"), "Workflow id")
        task_id = _name(workflow.get("taskId"), "Task id")
        _name(workflow.get("executionPipelineClass"), "Execution pipeline class")
        if workflow_id in workflow_ids or task_id in task_ids:
            raise ModularWorkflowContractError("Workflow and task ids must be unique.")
        workflow_ids.add(workflow_id)
        task_ids.add(task_id)
        if not isinstance(workflow.get("label"), str) or not workflow["label"] or len(workflow["label"]) > 256:
            raise ModularWorkflowContractError("Workflow label is invalid.")
        if workflow.get("kind") not in {"auto", "sequential", "loop", "block"}:
            raise ModularWorkflowContractError("Workflow kind is invalid.")
        for key in ("requiredInputs", "stateKeys"):
            names = workflow.get(key)
            if not isinstance(names, list) or len(names) > _MAX_FIELDS or len(set(names)) != len(names):
                raise ModularWorkflowContractError(f"Workflow {key} is malformed.")
            for name in names:
                _name(name, f"Workflow {key} name")
        for key in ("inputs", "outputs"):
            fields = workflow.get(key)
            if not isinstance(fields, list) or len(fields) > _MAX_FIELDS:
                raise ModularWorkflowContractError(f"Workflow {key} is malformed.")
            field_names = set()
            for field in fields:
                if not isinstance(field, dict) or set(field) != {
                    "name",
                    "type",
                    "required",
                    "default",
                    "description",
                }:
                    raise ModularWorkflowContractError(f"Workflow {key} field is malformed.")
                name = _name(field.get("name"), f"Workflow {key} name")
                if name in field_names or type(field.get("required")) is not bool:
                    raise ModularWorkflowContractError(f"Workflow {key} fields are duplicated or malformed.")
                field_names.add(name)
                if (
                    not isinstance(field.get("type"), str)
                    or len(field["type"]) > 256
                    or not isinstance(field.get("description"), str)
                    or len(field["description"]) > 1024
                ):
                    raise ModularWorkflowContractError(f"Workflow {key} field metadata is malformed.")
            if key == "inputs":
                all_input_names.update(field_names)
        if not set(workflow["requiredInputs"]).issubset({field["name"] for field in workflow["inputs"]}):
            raise ModularWorkflowContractError("Required workflow inputs must exist in the input contract.")
        alternatives = workflow.get("requiredInputAlternatives")
        if alternatives is not None:
            input_names = {field["name"] for field in workflow["inputs"]}
            if not isinstance(alternatives, list) or not 2 <= len(alternatives) <= _MAX_FIELDS:
                raise ModularWorkflowContractError("Required workflow input alternatives are malformed.")
            normalized_alternatives = []
            for alternative in alternatives:
                if (
                    not isinstance(alternative, list)
                    or not alternative
                    or len(alternative) > _MAX_FIELDS
                    or len(set(alternative)) != len(alternative)
                ):
                    raise ModularWorkflowContractError("Required workflow input alternatives are malformed.")
                for name in alternative:
                    _name(name, "Required workflow input alternative")
                if not set(alternative).issubset(input_names):
                    raise ModularWorkflowContractError("Required workflow input alternatives target unknown inputs.")
                normalized_alternatives.append(frozenset(alternative))
            if len(set(normalized_alternatives)) != len(normalized_alternatives):
                raise ModularWorkflowContractError("Required workflow input alternatives must be unique.")
            if set.intersection(*(set(item) for item in alternatives)) != set(workflow["requiredInputs"]):
                raise ModularWorkflowContractError(
                    "Required workflow inputs must equal the intersection of their alternatives."
                )
        steps = workflow.get("steps")
        if not isinstance(steps, list) or not steps or len(steps) > _MAX_STEPS:
            raise ModularWorkflowContractError("Workflow steps are malformed.")
        step_paths = set()
        for step in steps:
            if not isinstance(step, dict) or set(step) != {"path", "className", "kind", "description"}:
                raise ModularWorkflowContractError("Workflow step is malformed.")
            step_path = _name(step.get("path"), "Workflow step path")
            if step_path in step_paths:
                raise ModularWorkflowContractError("Workflow step paths must be unique.")
            step_paths.add(step_path)
            _name(step.get("className"), "Workflow step class")
            if step.get("kind") not in {"auto", "sequential", "loop", "block"}:
                raise ModularWorkflowContractError("Workflow step kind is invalid.")
            if not isinstance(step.get("description"), str) or len(step["description"]) > 1024:
                raise ModularWorkflowContractError("Workflow step description is invalid.")
    for name in normalized["uiDefaults"]:
        _name(name, "Workflow UI default name")
        if name not in all_input_names:
            raise ModularWorkflowContractError("Workflow UI defaults must target a published input.")
    return normalized


def select_modular_workflow(
    contract: Mapping[str, Any],
    task_id: str,
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve one exact normalized task and reject missing inputs."""

    normalized = validate_modular_workflow_contract(contract)
    workflow = next((item for item in normalized["workflows"] if item["taskId"] == task_id), None)
    if workflow is None:
        raise ModularWorkflowContractError(f"Unknown Modular workflow task {task_id!r}.")
    missing = sorted(name for name in workflow["requiredInputs"] if values.get(name) is None)
    if missing:
        raise ModularWorkflowContractError(
            f"Modular workflow task {task_id!r} is missing required inputs: {', '.join(missing)}."
        )
    alternatives = workflow.get("requiredInputAlternatives")
    if alternatives and not any(all(values.get(name) is not None for name in names) for names in alternatives):
        choices = " or ".join(" + ".join(names) for names in alternatives)
        raise ModularWorkflowContractError(
            f"Modular workflow task {task_id!r} requires one complete input set: {choices}."
        )
    return workflow


def validate_modular_workflow_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"schemaVersion", "diffusersRevision", "contracts"}:
        raise ModularWorkflowContractError("Modular workflow snapshot is malformed.")
    if value.get("schemaVersion") != 1 or value.get("diffusersRevision") != PINNED_DIFFUSERS_REVISION:
        raise ModularWorkflowContractError("Modular workflow snapshot does not match the pinned Diffusers revision.")
    contracts = value.get("contracts")
    if not isinstance(contracts, list) or not contracts or len(contracts) > _MAX_CONTRACTS:
        raise ModularWorkflowContractError("Modular workflow snapshot contracts are malformed.")
    normalized = [validate_modular_workflow_contract(contract) for contract in contracts]
    names = [contract["pipelineClass"] for contract in normalized]
    if len(set(names)) != len(names):
        raise ModularWorkflowContractError("Modular workflow snapshot pipeline classes must be unique.")
    return {
        "schemaVersion": 1,
        "diffusersRevision": PINNED_DIFFUSERS_REVISION,
        "contracts": normalized,
    }


def load_reviewed_modular_workflow_snapshot(path: Path = MODULAR_WORKFLOW_SNAPSHOT) -> dict[str, Any]:
    try:
        raw_bytes = path.read_bytes()
    except OSError as error:
        raise ModularWorkflowContractError(f"Could not read the reviewed Modular workflow snapshot: {error}") from error
    if len(raw_bytes) > 4 * 1024 * 1024:
        raise ModularWorkflowContractError("Reviewed Modular workflow snapshot exceeds the 4 MiB limit.")
    try:
        value = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise ModularWorkflowContractError(f"Reviewed Modular workflow snapshot is invalid JSON: {error}") from error
    return validate_modular_workflow_snapshot(value)


@lru_cache(maxsize=1)
def _cached_reviewed_modular_workflow_snapshot() -> dict[str, Any]:
    return load_reviewed_modular_workflow_snapshot()


def reviewed_modular_workflow_contract(pipeline_class: str) -> dict[str, Any]:
    snapshot = _cached_reviewed_modular_workflow_snapshot()
    contract = next(
        (item for item in snapshot["contracts"] if item["pipelineClass"] == pipeline_class),
        None,
    )
    if contract is None:
        raise ModularWorkflowContractError(f"No reviewed workflow contract exists for {pipeline_class!r}.")
    return contract
