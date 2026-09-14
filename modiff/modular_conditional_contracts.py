"""Pinned, no-weight contracts for unpruned Modular Diffusers block trees.

``ModularPipelineBlocks.get_workflow()`` intentionally returns an execution
tree whose conditional blocks have already been resolved.  That is ideal for
execution, but it is insufficient for an editor that must show the alternative
official branches and remember which trigger inputs selected one of them.

This companion snapshot keeps the original ``pipeline.blocks`` tree, the
presence/absence truth table for every ``ConditionalPipelineBlocks`` instance,
and an execution trace for every advertised workflow predicate.  Runtime
readers only parse reviewed JSON and never import Diffusers or model weights.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from modiff.modular_block_contracts import (
    _block_definition,
    _content_hash,
    _name,
    _validate_block_definition,
    load_reviewed_modular_block_snapshot,
)
from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION
from modiff.modular_workflow_discovery import load_reviewed_modular_workflow_snapshot


MODULAR_CONDITIONAL_CONTRACT_SCHEMA_VERSION = 1
MODULAR_CONDITIONAL_CONTRACT_SNAPSHOT = (
    Path(__file__).resolve().parents[1] / "data" / "modular-conditional-contracts.json"
)

_MAX_SNAPSHOT_BYTES = 48 * 1024 * 1024
_MAX_PIPELINES = 128
_MAX_DEFINITIONS = 2048
_MAX_PLACEMENTS = 4096
_MAX_CONDITIONALS = 512
_MAX_WORKFLOWS = 256
_MAX_CASES = 16
_MAX_TRIGGERS = 8
_MAX_PATH_DEPTH = 64


class ModularConditionalContractError(ValueError):
    """An unpruned Modular Diffusers conditional contract is malformed."""


def _mro_names(block: Any) -> set[str]:
    return {base.__name__ for base in type(block).__mro__}


def _is_conditional(block: Any) -> bool:
    return "ConditionalPipelineBlocks" in _mro_names(block)


def _is_loop(block: Any) -> bool:
    return "LoopSequentialPipelineBlocks" in _mro_names(block)


def _selection_outcome(block: Any, present_inputs: Sequence[str]) -> tuple[str | None, str | None]:
    present = set(present_inputs)
    trigger_inputs = [item for item in getattr(block, "block_trigger_inputs", ()) if item is not None]
    try:
        selected = block.select_block(**{name: True if name in present else None for name in trigger_inputs})
    except Exception as error:
        message = " ".join(str(error).split())[:512] or type(error).__name__
        return None, message
    if selected is None:
        selected = getattr(block, "default_block_name", None)
    if selected is not None and selected not in getattr(block, "sub_blocks", {}):
        raise ModularConditionalContractError(
            f"Conditional block {type(block).__name__} selected unknown branch {selected!r}."
        )
    return selected, None


def _effective_selection(block: Any, present_inputs: Sequence[str]) -> str | None:
    selected, error = _selection_outcome(block, present_inputs)
    if error is not None:
        raise ModularConditionalContractError(
            f"Conditional block {type(block).__name__} rejected its active inputs: {error}"
        )
    return selected


def _conditional_contract(block: Any, path: tuple[str, ...]) -> dict[str, Any]:
    raw_triggers = list(getattr(block, "block_trigger_inputs", ()))
    if len(raw_triggers) > _MAX_TRIGGERS or any(item is not None and not isinstance(item, str) for item in raw_triggers):
        raise ModularConditionalContractError("Conditional trigger inputs are malformed or oversized.")
    triggers = list(dict.fromkeys(item for item in raw_triggers if item is not None))
    branch_names = [_name(name, "Conditional branch name") for name in getattr(block, "sub_blocks", {})]
    if not branch_names:
        raise ModularConditionalContractError("A conditional block must expose at least one branch.")
    table = []
    for mask in itertools.product((False, True), repeat=len(triggers)):
        present = [name for name, enabled in zip(triggers, mask) if enabled]
        selected, error = _selection_outcome(block, present)
        table.append(
            {
                "presentInputs": present,
                "selectedBlockName": selected,
                "error": error,
            }
        )
    return {
        "path": list(path),
        "legacyPath": ".".join(path),
        "blockDefinitionId": _block_definition(block)["id"],
        "strategy": "auto" if "AutoPipelineBlocks" in _mro_names(block) else "presence",
        "branchNames": branch_names,
        "triggerInputs": raw_triggers,
        "defaultBlockName": getattr(block, "default_block_name", None),
        "selectionTable": table,
    }


def _workflow_requirement_cases(value: Any, workflow_id: str) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        candidates = [value]
    elif isinstance(value, tuple) and value and all(isinstance(item, Mapping) for item in value):
        candidates = list(value)
    else:
        raise ModularConditionalContractError(
            f"Workflow {workflow_id!r} has an unsupported upstream requirement predicate."
        )
    cases = []
    for candidate in candidates:
        if any(not isinstance(name, str) for name in candidate):
            raise ModularConditionalContractError(f"Workflow {workflow_id!r} predicate is malformed.")
        cases.append({"presentInputs": sorted(name for name, value in candidate.items() if value is not None)})
    return cases


def _trace_execution(blocks: Any, present_inputs: Sequence[str]) -> dict[str, Any]:
    active_inputs = {name: True for name in present_inputs}
    selections: list[dict[str, Any]] = []
    leaves: list[tuple[tuple[str, ...], Any]] = []

    def visit(block: Any, path: tuple[str, ...]) -> None:
        if _is_conditional(block):
            selected = _effective_selection(block, [name for name, value in active_inputs.items() if value is not None])
            selections.append(
                {
                    "path": list(path),
                    "legacyPath": ".".join(path),
                    "selectedBlockName": selected,
                }
            )
            if selected is None:
                return
            visit(block.sub_blocks[selected], (*path, selected))
            return
        nested = getattr(block, "sub_blocks", None)
        if isinstance(nested, Mapping) and nested and not _is_loop(block):
            for name, child in nested.items():
                visit(child, (*path, _name(name, "Execution block name")))
            return
        leaves.append((path, block))
        for output in getattr(block, "intermediate_outputs", ()):
            name = getattr(output, "name", None)
            if isinstance(name, str) and name:
                active_inputs[name] = True

    for name, block in blocks.sub_blocks.items():
        visit(block, (_name(name, "Top-level block name"),))
    return {
        "selections": selections,
        "activeLeafPaths": [list(path) for path, _block in leaves],
        "activeLeafDefinitionIds": [_block_definition(block)["id"] for _path, block in leaves],
    }


def build_modular_conditional_contract(pipeline: Any) -> dict[str, Any]:
    """Discover one original, unpruned no-weight Modular block tree."""

    pipeline_class = _name(type(pipeline).__name__, "Pipeline class")
    blocks = pipeline.blocks
    root_definition = _block_definition(blocks)
    definitions = {root_definition["id"]: root_definition}
    placements: list[dict[str, Any]] = []
    conditionals: list[dict[str, Any]] = []

    def visit(block: Any, path: tuple[str, ...], order: int) -> None:
        if len(path) > _MAX_PATH_DEPTH or len(placements) >= _MAX_PLACEMENTS:
            raise ModularConditionalContractError("Unpruned block tree exceeds its placement limit.")
        definition = _block_definition(block)
        previous = definitions.get(definition["id"])
        if previous is not None and previous != definition:
            raise ModularConditionalContractError("A static block id resolved to inconsistent definitions.")
        definitions[definition["id"]] = definition
        placements.append(
            {
                "path": list(path),
                "legacyPath": ".".join(path),
                "order": order,
                "blockDefinitionId": definition["id"],
                "intermediateOutputs": [
                    _name(output.name, "Conditional intermediate output")
                    for output in getattr(block, "intermediate_outputs", ())
                    if isinstance(getattr(output, "name", None), str)
                ],
            }
        )
        if _is_conditional(block):
            conditionals.append(_conditional_contract(block, path))
        nested = getattr(block, "sub_blocks", None)
        if isinstance(nested, Mapping):
            for child_order, (name, child) in enumerate(nested.items()):
                visit(child, (*path, _name(name, "Sub-block name")), child_order)

    for order, (name, block) in enumerate(blocks.sub_blocks.items()):
        visit(block, (_name(name, "Top-level block name"),), order)

    raw_map = getattr(blocks, "_workflow_map", None)
    if isinstance(raw_map, Mapping) and raw_map:
        workflow_ids = list(blocks.available_workflows)
        if set(workflow_ids) != set(raw_map):
            raise ModularConditionalContractError("available_workflows disagrees with the upstream workflow map.")
        predicates = raw_map
    else:
        workflow_ids = ["default"]
        predicates = {"default": {}}
    workflows = []
    for workflow_id_value in workflow_ids:
        workflow_id = _name(workflow_id_value, "Workflow id")
        cases = []
        for requirement_case in _workflow_requirement_cases(predicates[workflow_id], workflow_id):
            trace = _trace_execution(blocks, requirement_case["presentInputs"])
            cases.append({**requirement_case, **trace})
        workflows.append({"id": workflow_id, "cases": cases})

    body = {
        "schemaVersion": MODULAR_CONDITIONAL_CONTRACT_SCHEMA_VERSION,
        "pipelineClass": pipeline_class,
        "rootBlockDefinitionId": root_definition["id"],
        "placements": placements,
        "conditionals": conditionals,
        "workflows": workflows,
    }
    return {
        "blockDefinitions": sorted(definitions.values(), key=lambda item: item["id"]),
        "pipeline": {**body, "contentHash": _content_hash(body)},
    }


def merge_modular_conditional_contracts(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    definitions: dict[str, dict[str, Any]] = {}
    pipelines = []
    for part in parts:
        for definition in part.get("blockDefinitions", ()):
            previous = definitions.get(definition["id"])
            if previous is not None and previous != definition:
                raise ModularConditionalContractError("A shared static block id is inconsistent.")
            definitions[definition["id"]] = definition
        pipelines.append(part["pipeline"])
    return validate_modular_conditional_snapshot(
        {
            "schemaVersion": MODULAR_CONDITIONAL_CONTRACT_SCHEMA_VERSION,
            "diffusersRevision": PINNED_DIFFUSERS_REVISION,
            "blockDefinitions": sorted(definitions.values(), key=lambda item: item["id"]),
            "pipelines": sorted(pipelines, key=lambda item: item["pipelineClass"]),
        }
    )


def _checked_path(value: Any, label: str, *, allow_root: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > _MAX_PATH_DEPTH or (not allow_root and not value):
        raise ModularConditionalContractError(f"{label} is malformed.")
    return tuple(_name(segment, f"{label} segment") for segment in value)


def validate_modular_conditional_snapshot(value: Any) -> dict[str, Any]:
    """Validate, cross-check, and detach the complete unpruned snapshot."""

    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        normalized = json.loads(encoded)
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ModularConditionalContractError(f"Conditional snapshot is not finite JSON: {error}") from error
    if len(encoded.encode("utf-8")) > _MAX_SNAPSHOT_BYTES:
        raise ModularConditionalContractError("Conditional snapshot exceeds the 48 MiB limit.")
    if not isinstance(normalized, dict) or set(normalized) != {
        "schemaVersion",
        "diffusersRevision",
        "blockDefinitions",
        "pipelines",
    }:
        raise ModularConditionalContractError("Conditional snapshot has missing or unknown top-level fields.")
    if (
        normalized.get("schemaVersion") != MODULAR_CONDITIONAL_CONTRACT_SCHEMA_VERSION
        or normalized.get("diffusersRevision") != PINNED_DIFFUSERS_REVISION
    ):
        raise ModularConditionalContractError("Conditional snapshot does not match the pinned Diffusers revision.")
    definitions = normalized.get("blockDefinitions")
    pipelines = normalized.get("pipelines")
    if not isinstance(definitions, list) or not 0 < len(definitions) <= _MAX_DEFINITIONS:
        raise ModularConditionalContractError("Conditional block definitions are malformed.")
    if not isinstance(pipelines, list) or not 0 < len(pipelines) <= _MAX_PIPELINES:
        raise ModularConditionalContractError("Conditional pipeline collection is malformed.")

    # Reuse the stricter reviewed block validator for every definition by
    # temporarily placing it in a real reviewed workflow is not possible
    # because static-only alternatives are intentionally unreferenced there.
    # The content-derived identity is nevertheless fully rechecked here.
    definitions_by_id = {}
    for definition in definitions:
        try:
            validated_definition = _validate_block_definition(definition)
        except ValueError as error:
            raise ModularConditionalContractError(str(error)) from error
        definition_id = validated_definition["id"]
        if definition_id in definitions_by_id:
            raise ModularConditionalContractError("Conditional block definition ids must be unique.")
        definitions_by_id[definition_id] = validated_definition

    reviewed_workflows = load_reviewed_modular_workflow_snapshot()
    reviewed_blocks = load_reviewed_modular_block_snapshot()
    expected_workflows = {
        contract["pipelineClass"]: {workflow["id"] for workflow in contract["workflows"]}
        for contract in reviewed_workflows["contracts"]
    }
    resolved_by_key = {
        (workflow["pipelineClass"], workflow["workflowId"]): workflow
        for workflow in reviewed_blocks["workflows"]
    }
    resolved_definitions = {item["id"]: item for item in reviewed_blocks["blockDefinitions"]}
    pipeline_names = set()
    referenced = set()
    for pipeline in pipelines:
        required_keys = {
            "schemaVersion",
            "pipelineClass",
            "rootBlockDefinitionId",
            "placements",
            "conditionals",
            "workflows",
            "contentHash",
        }
        if not isinstance(pipeline, dict) or set(pipeline) != required_keys:
            raise ModularConditionalContractError("Conditional pipeline contract is malformed.")
        pipeline_class = _name(pipeline.get("pipelineClass"), "Conditional pipeline class")
        if pipeline_class in pipeline_names:
            raise ModularConditionalContractError("Conditional pipeline classes must be unique.")
        pipeline_names.add(pipeline_class)
        root_id = pipeline.get("rootBlockDefinitionId")
        if root_id not in definitions_by_id:
            raise ModularConditionalContractError("Conditional pipeline root definition is unknown.")
        referenced.add(root_id)
        placements = pipeline.get("placements")
        if not isinstance(placements, list) or not placements or len(placements) > _MAX_PLACEMENTS:
            raise ModularConditionalContractError("Conditional placements are malformed.")
        paths = {}
        orders_by_parent: dict[tuple[str, ...], list[int]] = {}
        for placement in placements:
            if not isinstance(placement, dict) or set(placement) != {
                "path",
                "legacyPath",
                "order",
                "blockDefinitionId",
                "intermediateOutputs",
            }:
                raise ModularConditionalContractError("Conditional placement is malformed.")
            path = _checked_path(placement.get("path"), "Conditional placement path")
            if path in paths or placement.get("legacyPath") != ".".join(path):
                raise ModularConditionalContractError("Conditional placement path is duplicated or inconsistent.")
            order = placement.get("order")
            if not isinstance(order, int) or isinstance(order, bool) or order < 0:
                raise ModularConditionalContractError("Conditional placement order is malformed.")
            definition_id = placement.get("blockDefinitionId")
            if definition_id not in definitions_by_id:
                raise ModularConditionalContractError("Conditional placement references an unknown definition.")
            intermediate_outputs = placement.get("intermediateOutputs")
            if (
                not isinstance(intermediate_outputs, list)
                or len(intermediate_outputs) != len(set(intermediate_outputs))
                or any(not isinstance(name, str) or not name for name in intermediate_outputs)
            ):
                raise ModularConditionalContractError("Conditional placement intermediate outputs are malformed.")
            paths[path] = definition_id
            referenced.add(definition_id)
            orders_by_parent.setdefault(path[:-1], []).append(order)
        if any(len(path) > 1 and path[:-1] not in paths for path in paths):
            raise ModularConditionalContractError("Conditional placement has a missing parent.")
        if any(sorted(orders) != list(range(len(orders))) for orders in orders_by_parent.values()):
            raise ModularConditionalContractError("Conditional sibling order is not contiguous.")

        conditional_paths = set()
        conditionals = pipeline.get("conditionals")
        if not isinstance(conditionals, list) or len(conditionals) > _MAX_CONDITIONALS:
            raise ModularConditionalContractError("Conditional selector collection is malformed.")
        for conditional in conditionals:
            if not isinstance(conditional, dict) or set(conditional) != {
                "path",
                "legacyPath",
                "blockDefinitionId",
                "strategy",
                "branchNames",
                "triggerInputs",
                "defaultBlockName",
                "selectionTable",
            }:
                raise ModularConditionalContractError("Conditional selector is malformed.")
            path = _checked_path(conditional.get("path"), "Conditional selector path", allow_root=True)
            expected_definition_id = root_id if not path else paths.get(path)
            if conditional.get("blockDefinitionId") != expected_definition_id:
                raise ModularConditionalContractError("Conditional selector definition is inconsistent.")
            if path in conditional_paths or conditional.get("legacyPath") != ".".join(path):
                raise ModularConditionalContractError("Conditional selector path is duplicated or inconsistent.")
            conditional_paths.add(path)
            if conditional.get("strategy") not in {"auto", "presence"}:
                raise ModularConditionalContractError("Conditional selector strategy is invalid.")
            branches = conditional.get("branchNames")
            triggers = conditional.get("triggerInputs")
            if (
                not isinstance(branches, list)
                or not branches
                or len(branches) != len(set(branches))
                or any(not isinstance(name, str) for name in branches)
                or not isinstance(triggers, list)
                or len(triggers) > _MAX_TRIGGERS
                or any(name is not None and not isinstance(name, str) for name in triggers)
            ):
                raise ModularConditionalContractError("Conditional branches or triggers are malformed.")
            if set(branches) != {candidate[-1] for candidate in paths if candidate[:-1] == path}:
                raise ModularConditionalContractError("Conditional branches disagree with the static tree.")
            default = conditional.get("defaultBlockName")
            if default is not None and default not in branches:
                raise ModularConditionalContractError("Conditional default branch is unknown.")
            unique_triggers = list(dict.fromkeys(name for name in triggers if name is not None))
            table = conditional.get("selectionTable")
            if not isinstance(table, list) or len(table) != 2 ** len(unique_triggers):
                raise ModularConditionalContractError("Conditional truth table is incomplete.")
            expected_presence = {
                tuple(name for name, enabled in zip(unique_triggers, mask) if enabled)
                for mask in itertools.product((False, True), repeat=len(unique_triggers))
            }
            actual_presence = set()
            for row in table:
                if not isinstance(row, dict) or set(row) != {"presentInputs", "selectedBlockName", "error"}:
                    raise ModularConditionalContractError("Conditional truth-table row is malformed.")
                present = tuple(row.get("presentInputs", ()))
                if any(name not in unique_triggers for name in present):
                    raise ModularConditionalContractError("Conditional truth-table input is unknown.")
                selected = row.get("selectedBlockName")
                if selected is not None and selected not in branches:
                    raise ModularConditionalContractError("Conditional truth-table branch is unknown.")
                error = row.get("error")
                if error is not None and (not isinstance(error, str) or not error or len(error) > 512):
                    raise ModularConditionalContractError("Conditional truth-table error is malformed.")
                if error is not None and selected is not None:
                    raise ModularConditionalContractError("A rejected conditional input cannot select a branch.")
                actual_presence.add(present)
            if actual_presence != expected_presence:
                raise ModularConditionalContractError("Conditional truth-table coverage is invalid.")

        workflows = pipeline.get("workflows")
        if not isinstance(workflows, list) or not workflows or len(workflows) > _MAX_WORKFLOWS:
            raise ModularConditionalContractError("Conditional workflows are malformed.")
        workflow_ids = set()
        for workflow in workflows:
            if not isinstance(workflow, dict) or set(workflow) != {"id", "cases"}:
                raise ModularConditionalContractError("Conditional workflow is malformed.")
            workflow_id = _name(workflow.get("id"), "Conditional workflow id")
            if workflow_id in workflow_ids:
                raise ModularConditionalContractError("Conditional workflow ids must be unique.")
            workflow_ids.add(workflow_id)
            cases = workflow.get("cases")
            if not isinstance(cases, list) or not cases or len(cases) > _MAX_CASES:
                raise ModularConditionalContractError("Conditional workflow cases are malformed.")
            resolved = resolved_by_key.get((pipeline_class, workflow_id))
            if resolved is None:
                raise ModularConditionalContractError("Conditional workflow is absent from the resolved snapshot.")
            resolved_kind_by_path = {
                tuple(placement["path"]): resolved_definitions[placement["blockDefinitionId"]]["kind"]
                for placement in resolved["placements"]
            }
            expected_leaf_classes = []
            for placement in resolved["placements"]:
                path = tuple(placement["path"])
                kind = resolved_kind_by_path[path]
                inside_loop = any(
                    resolved_kind_by_path.get(path[:depth]) == "loop" for depth in range(1, len(path))
                )
                if kind in {"block", "loop"} and not inside_loop:
                    expected_leaf_classes.append(
                        resolved_definitions[placement["blockDefinitionId"]]["className"]
                    )
            for case in cases:
                if not isinstance(case, dict) or set(case) != {
                    "presentInputs",
                    "selections",
                    "activeLeafPaths",
                    "activeLeafDefinitionIds",
                }:
                    raise ModularConditionalContractError("Conditional workflow case is malformed.")
                if not isinstance(case["presentInputs"], list) or any(
                    not isinstance(name, str) for name in case["presentInputs"]
                ):
                    raise ModularConditionalContractError("Conditional workflow inputs are malformed.")
                active_paths = [_checked_path(path, "Active leaf path") for path in case["activeLeafPaths"]]
                if any(path not in paths for path in active_paths):
                    raise ModularConditionalContractError("Conditional workflow references an unknown active leaf.")
                active_ids = case["activeLeafDefinitionIds"]
                active_classes = [definitions_by_id[definition_id]["className"] for definition_id in active_ids]
                if (
                    active_ids != [paths[path] for path in active_paths]
                    or active_classes != expected_leaf_classes
                ):
                    raise ModularConditionalContractError(
                        "Conditional workflow leaf sequence disagrees with the resolved reviewed workflow "
                        f"for {pipeline_class}/{workflow_id}."
                    )
                if not isinstance(case["selections"], list):
                    raise ModularConditionalContractError("Conditional workflow selections are malformed.")
                for selection in case["selections"]:
                    if not isinstance(selection, dict) or set(selection) != {
                        "path",
                        "legacyPath",
                        "selectedBlockName",
                    }:
                        raise ModularConditionalContractError("Conditional workflow selection is malformed.")
                    selection_path = _checked_path(
                        selection["path"], "Conditional workflow selection path", allow_root=True
                    )
                    if selection_path not in conditional_paths or selection["legacyPath"] != ".".join(selection_path):
                        raise ModularConditionalContractError("Conditional workflow selection path is unknown.")
        if workflow_ids != expected_workflows.get(pipeline_class):
            raise ModularConditionalContractError("Conditional workflows do not match reviewed workflow coverage.")
        body = {key: item for key, item in pipeline.items() if key != "contentHash"}
        if pipeline.get("contentHash") != _content_hash(body):
            raise ModularConditionalContractError("Conditional pipeline content hash is invalid.")

    if pipeline_names != set(expected_workflows):
        raise ModularConditionalContractError("Conditional pipelines do not exactly cover reviewed pipelines.")
    if referenced != set(definitions_by_id):
        raise ModularConditionalContractError("Conditional snapshot contains unreferenced definitions.")
    return normalized


@lru_cache(maxsize=1)
def load_reviewed_modular_conditional_snapshot(
    path: Path = MODULAR_CONDITIONAL_CONTRACT_SNAPSHOT,
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ModularConditionalContractError(f"Cannot load conditional snapshot: {error}") from error
    return validate_modular_conditional_snapshot(value)


def reviewed_modular_conditional_snapshot() -> dict[str, Any]:
    return json.loads(json.dumps(load_reviewed_modular_conditional_snapshot(), allow_nan=False))


def select_conditional_branch(conditional: Mapping[str, Any], present_inputs: Sequence[str]) -> str | None:
    """Resolve one reviewed truth table without importing Diffusers."""

    present = set(present_inputs)
    triggers = list(dict.fromkeys(name for name in conditional.get("triggerInputs", ()) if name is not None))
    key = [name for name in triggers if name in present]
    for row in conditional.get("selectionTable", ()):
        if row.get("presentInputs") == key:
            if row.get("error") is not None:
                raise ModularConditionalContractError(str(row["error"]))
            return row.get("selectedBlockName")
    raise ModularConditionalContractError("No reviewed conditional truth-table row matches these inputs.")
