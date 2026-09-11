"""Pinned, fail-closed composition recipes for official Modular Diffusers blocks.

The visual editor may compose ordinary MoDiff nodes freely inside a User Node.
That graph editing is deliberately distinct from changing an upstream
``ModularPipelineBlocks`` tree. This module owns the latter boundary: a recipe
starts from one reviewed pipeline/workflow at the pinned Diffusers commit,
allows exact reviewed source blocks and authenticated structural edits, and
rebuilds the modified unpruned tree through upstream ``init_pipeline()``.

No model weights are loaded here. The returned receipt is a structural rebuild
receipt, not model/runtime admission and not an executable publication claim.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from modiff.huggingface_node_library import reviewed_huggingface_node_library
from modiff.modular_conditional_contracts import reviewed_modular_conditional_snapshot
from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION


MODULAR_COMPOSITION_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,255}$")
_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_OPERATIONS = 128
_MAX_PATH_DEPTH = 64


class ModularCompositionError(ValueError):
    """A requested upstream block composition is unknown or incompatible."""


def _hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ModularCompositionError(f"{label} must be a bounded identifier.")
    return value


def _path(value: Any, label: str, *, allow_root: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > _MAX_PATH_DEPTH or (not value and not allow_root):
        raise ModularCompositionError(f"{label} must be a bounded block path.")
    return tuple(_identifier(segment, f"{label} segment") for segment in value)


def _definition_for_recipe(library: Mapping[str, Any], pipeline_class: str, workflow_id: str) -> dict[str, Any]:
    matches = [
        item
        for item in library.get("definitions", ())
        if item.get("provider") == "diffusers"
        and item.get("pipelineClass") == pipeline_class
        and item.get("workflowId") == workflow_id
    ]
    if len(matches) != 1:
        raise ModularCompositionError(
            f"Reviewed Modular workflow {pipeline_class}/{workflow_id} does not resolve to one exact definition."
        )
    return matches[0]


def _definition_by_id(library: Mapping[str, Any], definition_id: str) -> dict[str, Any]:
    matches = [item for item in library.get("definitions", ()) if item.get("id") == definition_id]
    if len(matches) != 1:
        raise ModularCompositionError(f"Reviewed Modular source definition {definition_id!r} is unavailable.")
    return matches[0]


def _block_definitions(library: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["id"]: item
        for item in library.get("blockDefinitions", ())
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].startswith("diffusers.")
    }


def _unpruned_pipeline(pipeline_class: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    snapshot = reviewed_modular_conditional_snapshot()
    if snapshot.get("diffusersRevision") != PINNED_DIFFUSERS_REVISION:
        raise ModularCompositionError("The reviewed unpruned block tree does not match the pinned Diffusers commit.")
    matches = [item for item in snapshot.get("pipelines", ()) if item.get("pipelineClass") == pipeline_class]
    if len(matches) != 1:
        raise ModularCompositionError(f"No unique reviewed unpruned block tree exists for {pipeline_class!r}.")
    definitions = {
        item["id"]: item
        for item in snapshot.get("blockDefinitions", ())
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    return matches[0], definitions


def _descendants(paths: Mapping[tuple[str, ...], str], prefix: tuple[str, ...]) -> list[tuple[str, ...]]:
    return [path for path in paths if path[: len(prefix)] == prefix]


def _require_destination(
    paths: Mapping[tuple[str, ...], str],
    block_definitions: Mapping[str, Mapping[str, Any]],
    parent: tuple[str, ...],
) -> None:
    if not parent:
        return
    definition_id = paths.get(parent)
    definition = block_definitions.get(definition_id or "")
    if definition is None or definition.get("kind") not in {"sequential", "loop", "auto"}:
        raise ModularCompositionError(f"Destination block {'.'.join(parent)!r} is not a reviewed container.")


def _sibling_count(paths: Mapping[tuple[str, ...], str], parent: tuple[str, ...]) -> int:
    return sum(1 for path in paths if path[:-1] == parent)


def validate_modular_composition_recipe(
    value: Any,
    *,
    library: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and simulate one immutable reviewed block-tree recipe."""

    try:
        normalized = json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ModularCompositionError(f"Modular composition recipe is not finite JSON: {error}") from error
    if not isinstance(normalized, dict) or set(normalized) != {
        "schemaVersion",
        "diffusersRevision",
        "pipelineClass",
        "workflowId",
        "definitionId",
        "blockContractHash",
        "operations",
    }:
        raise ModularCompositionError("Modular composition recipe has missing or unknown fields.")
    if normalized.get("schemaVersion") != MODULAR_COMPOSITION_SCHEMA_VERSION:
        raise ModularCompositionError("Modular composition recipe requires schemaVersion 1.")
    if normalized.get("diffusersRevision") != PINNED_DIFFUSERS_REVISION:
        raise ModularCompositionError("Modular composition recipe does not use MoDiff's pinned Diffusers commit.")
    pipeline_class = _identifier(normalized.get("pipelineClass"), "Pipeline class")
    workflow_id = _identifier(normalized.get("workflowId"), "Workflow id")
    definition_id = normalized.get("definitionId")
    contract_hash = normalized.get("blockContractHash")
    if not isinstance(definition_id, str) or not definition_id.startswith("diffusers.modular:"):
        raise ModularCompositionError("Modular composition definition id is invalid.")
    if not isinstance(contract_hash, str) or _CONTENT_HASH.fullmatch(contract_hash) is None:
        raise ModularCompositionError("Modular composition block contract hash is invalid.")
    operations = normalized.get("operations")
    if not isinstance(operations, list) or len(operations) > _MAX_OPERATIONS:
        raise ModularCompositionError("Modular composition operations are malformed or oversized.")

    reviewed = library if library is not None else reviewed_huggingface_node_library()
    if reviewed.get("diffusersRevision") != PINNED_DIFFUSERS_REVISION:
        raise ModularCompositionError("The reviewed node library does not match the pinned Diffusers commit.")
    definition = _definition_for_recipe(reviewed, pipeline_class, workflow_id)
    if definition.get("id") != definition_id or definition.get("blockContractHash") != contract_hash:
        raise ModularCompositionError("The recipe definition or block contract no longer matches the reviewed library.")
    unpruned, unpruned_definitions = _unpruned_pipeline(pipeline_class)
    definitions = {**_block_definitions(reviewed), **unpruned_definitions}
    paths = {
        tuple(placement["path"]): placement["blockDefinitionId"] for placement in unpruned["placements"]
    }
    original_paths = dict(paths)
    normalized_operations = []

    for index, operation in enumerate(operations):
        if not isinstance(operation, dict) or operation.get("kind") not in {
            "remove",
            "move",
            "duplicate",
            "insert",
            "replace",
        }:
            raise ModularCompositionError(f"Composition operation {index + 1} has an unsupported kind.")
        kind = operation["kind"]
        if kind == "remove":
            expected_keys = {"kind", "path"}
        elif kind in {"move", "duplicate"}:
            expected_keys = {"kind", "path", "parentPath", "name", "index"}
        elif kind == "insert":
            expected_keys = {
                "kind",
                "sourceDefinitionId",
                "sourceBlockDefinitionId",
                "sourcePath",
                "sourceExecutionScope",
                "parentPath",
                "name",
                "index",
            }
        else:
            expected_keys = {
                "kind",
                "path",
                "sourceDefinitionId",
                "sourceBlockDefinitionId",
                "sourcePath",
                "sourceExecutionScope",
            }
        if set(operation) != expected_keys:
            raise ModularCompositionError(f"Composition operation {index + 1} has missing or unknown fields.")
        source_path = None
        if kind != "insert":
            source_path = _path(operation.get("path"), f"Operation {index + 1} path")
            if source_path not in paths:
                raise ModularCompositionError(f"Composition source {'.'.join(source_path)!r} is not present.")
        if kind == "remove":
            for path in _descendants(paths, source_path):
                paths.pop(path)
            normalized_operations.append({"kind": kind, "path": list(source_path)})
            continue

        if kind in {"insert", "replace"}:
            source_definition_id = operation.get("sourceDefinitionId")
            source_block_definition_id = operation.get("sourceBlockDefinitionId")
            source_execution_scope = operation.get("sourceExecutionScope")
            if not isinstance(source_definition_id, str) or not source_definition_id.startswith("diffusers.modular:"):
                raise ModularCompositionError(f"Operation {index + 1} source definition id is invalid.")
            if not isinstance(source_block_definition_id, str) or source_block_definition_id not in definitions:
                raise ModularCompositionError(f"Operation {index + 1} source block definition is unavailable.")
            source_definition = _definition_by_id(reviewed, source_definition_id)
            if source_definition.get("provider") != "diffusers":
                raise ModularCompositionError(
                    f"Operation {index + 1} must use a reviewed Diffusers block."
                )
            reviewed_source_path = _path(operation.get("sourcePath"), f"Operation {index + 1} source path")
            if source_execution_scope == "selected_workflow":
                source_placements = {
                    tuple(placement["path"]): placement["blockDefinitionId"]
                    for placement in source_definition.get("blockPlacements", ())
                }
            elif source_execution_scope == "unpruned_pipeline":
                source_pipeline, source_definitions = _unpruned_pipeline(source_definition["pipelineClass"])
                definitions.update(source_definitions)
                source_placements = {
                    tuple(placement["path"]): placement["blockDefinitionId"]
                    for placement in source_pipeline["placements"]
                }
            else:
                raise ModularCompositionError(f"Operation {index + 1} source execution scope is unsupported.")
            if source_placements.get(reviewed_source_path) != source_block_definition_id:
                raise ModularCompositionError(
                    f"Operation {index + 1} source path does not match its pinned block definition."
                )
            source_subtree = {
                path: definition_id_at_path
                for path, definition_id_at_path in source_placements.items()
                if path[: len(reviewed_source_path)] == reviewed_source_path
            }
            if not source_subtree:
                raise ModularCompositionError(f"Operation {index + 1} source subtree is unavailable.")
            # This validates editable structure and exact source authority, not
            # model readiness. Original-tree declarations cannot reject a draft
            # whose components may have been replaced. Execution checks the
            # actual upstream ComponentSpecs against the actual owned bundle.
            if kind == "insert":
                parent = _path(operation.get("parentPath"), f"Operation {index + 1} parent path", allow_root=True)
                name = _identifier(operation.get("name"), f"Operation {index + 1} block name")
                insertion_index = operation.get("index")
                if not isinstance(insertion_index, int) or isinstance(insertion_index, bool) or insertion_index < 0:
                    raise ModularCompositionError(f"Operation {index + 1} insertion index is invalid.")
                _require_destination(paths, definitions, parent)
                target_path = (*parent, name)
                if target_path in paths:
                    raise ModularCompositionError(f"Composition destination {'.'.join(target_path)!r} already exists.")
                if insertion_index > _sibling_count(paths, parent):
                    raise ModularCompositionError(
                        f"Operation {index + 1} insertion index exceeds the destination container size."
                    )
            else:
                target_path = source_path
                parent = target_path[:-1]
                name = target_path[-1]
                insertion_index = sum(
                    1
                    for candidate in paths
                    if candidate[:-1] == parent and candidate != target_path and candidate[-1] < target_path[-1]
                )
                for path in _descendants(paths, target_path):
                    paths.pop(path)
            additions = {
                (*target_path, *path[len(reviewed_source_path) :]): definition_id_at_path
                for path, definition_id_at_path in source_subtree.items()
            }
            if any(path in paths for path in additions):
                raise ModularCompositionError("The Modular block subtree would overwrite another reviewed block.")
            paths.update(additions)
            normalized_operation = {
                "kind": kind,
                "sourceDefinitionId": source_definition_id,
                "sourceBlockDefinitionId": source_block_definition_id,
                "sourcePath": list(reviewed_source_path),
                "sourceExecutionScope": source_execution_scope,
            }
            if kind == "insert":
                normalized_operation.update(
                    {
                        "parentPath": list(parent),
                        "name": name,
                        "index": insertion_index,
                    }
                )
            else:
                normalized_operation["path"] = list(target_path)
            normalized_operations.append(normalized_operation)
            continue

        parent = _path(operation.get("parentPath"), f"Operation {index + 1} parent path", allow_root=True)
        name = _identifier(operation.get("name"), f"Operation {index + 1} block name")
        insertion_index = operation.get("index")
        if not isinstance(insertion_index, int) or isinstance(insertion_index, bool) or insertion_index < 0:
            raise ModularCompositionError(f"Operation {index + 1} insertion index is invalid.")
        _require_destination(paths, definitions, parent)
        if kind == "move" and parent[: len(source_path)] == source_path:
            raise ModularCompositionError("A Modular block cannot be moved into itself or its descendant.")
        target_path = (*parent, name)
        target_collision = target_path in paths and not (kind == "move" and target_path == source_path)
        if target_collision:
            raise ModularCompositionError(f"Composition destination {'.'.join(target_path)!r} already exists.")
        destination_count = _sibling_count(paths, parent)
        if kind == "move" and source_path[:-1] == parent:
            destination_count -= 1
        if insertion_index > destination_count:
            raise ModularCompositionError(
                f"Operation {index + 1} insertion index exceeds the destination container size."
            )
        source_paths = _descendants(paths, source_path)
        additions = {
            (*target_path, *path[len(source_path) :]): paths[path]
            for path in source_paths
        }
        if any(path in paths and path not in source_paths for path in additions):
            raise ModularCompositionError("The Modular block subtree would overwrite another reviewed block.")
        if kind == "move":
            for path in source_paths:
                paths.pop(path)
        paths.update(additions)
        normalized_operations.append(
            {
                "kind": kind,
                "path": list(source_path),
                "parentPath": list(parent),
                "name": name,
                "index": insertion_index,
            }
        )

    if not paths:
        raise ModularCompositionError("A Modular composition cannot remove every workflow block.")
    for path, definition_id_at_path in paths.items():
        if len(path) > 1 and path[:-1] not in paths:
            raise ModularCompositionError("A Modular composition leaves a block without its container.")
        parent = path[:-1]
        if parent:
            parent_definition = definitions.get(paths[parent])
            child_definition = definitions.get(definition_id_at_path)
            if parent_definition is None or child_definition is None:
                raise ModularCompositionError("A Modular composition references an unknown block contract.")
            if parent_definition.get("kind") == "loop" and child_definition.get("kind") != "block":
                raise ModularCompositionError("Upstream loop containers accept only reviewed leaf blocks.")

    body = {
        "schemaVersion": MODULAR_COMPOSITION_SCHEMA_VERSION,
        "diffusersRevision": PINNED_DIFFUSERS_REVISION,
        "pipelineClass": pipeline_class,
        "workflowId": workflow_id,
        "definitionId": definition_id,
        "blockContractHash": contract_hash,
        "operations": normalized_operations,
    }
    return {
        **body,
        "recipeHash": _hash(body),
        "originalPathCount": len(original_paths),
        "composedPathCount": len(paths),
        "composedPaths": [list(path) for path in sorted(paths)],
        "composedPlacements": [
            {"path": list(path), "blockDefinitionId": paths[path]} for path in sorted(paths)
        ],
    }


def _default_blocks_resolver(pipeline_class: str, workflow_id: str):
    diffusers = importlib.import_module("diffusers")
    pipeline_type = getattr(diffusers, pipeline_class, None)
    if not isinstance(pipeline_type, type):
        raise ModularCompositionError(f"Pinned Diffusers does not export pipeline class {pipeline_class!r}.")
    definition = pipeline_type()
    blocks = definition.blocks
    if workflow_id == "__unpruned__":
        return blocks
    if workflow_id != "default":
        blocks = blocks.get_workflow(workflow_id)
    return blocks


def _block_at_path(blocks: Any, path: tuple[str, ...]):
    current = blocks
    for segment in path:
        sub_blocks = getattr(current, "sub_blocks", None)
        if not isinstance(sub_blocks, Mapping) or segment not in sub_blocks:
            raise ModularCompositionError(f"Runtime block path {'.'.join(path)!r} is unavailable.")
        current = sub_blocks[segment]
    return current


def _insert(sub_blocks: Any, name: str, block: Any, index: int) -> None:
    insert = getattr(sub_blocks, "insert", None)
    if not callable(insert):
        raise ModularCompositionError("The pinned Diffusers block container is not insertable.")
    insert(name, block, index)


def _field_names(values: Any) -> list[str]:
    return sorted(
        {
            str(getattr(value, "name"))
            for value in values or ()
            if isinstance(getattr(value, "name", None), str) and getattr(value, "name")
        }
    )


def build_reviewed_modular_composition_blocks(
    value: Any,
    *,
    library: Mapping[str, Any] | None = None,
    blocks_resolver: Callable[[str, str], Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    """Return the validated edited tree; the caller owns pipeline initialization.

    The model executor uses the same reviewed operations as structural preview,
    but supplies its existing ComponentsManager and retains nested runtime paths.
    No weights, arbitrary Python imports, or serialized runtime objects enter here.
    """

    reviewed = library if library is not None else reviewed_huggingface_node_library()
    recipe = validate_modular_composition_recipe(value, library=reviewed)
    _definition_for_recipe(reviewed, recipe["pipelineClass"], recipe["workflowId"])
    unpruned, unpruned_definitions = _unpruned_pipeline(recipe["pipelineClass"])
    definitions = {**_block_definitions(reviewed), **unpruned_definitions}
    expected_classes = {
        tuple(placement["path"]): definitions[placement["blockDefinitionId"]]["className"]
        for placement in unpruned["placements"]
    }
    resolver = blocks_resolver or _default_blocks_resolver
    blocks = resolver(recipe["pipelineClass"], "__unpruned__")
    for path, expected_class in expected_classes.items():
        current = _block_at_path(blocks, path)
        if type(current).__name__ != expected_class:
            raise ModularCompositionError(
                f"Runtime block {'.'.join(path)!r} is {type(current).__name__}, not reviewed {expected_class}."
            )

    source_blocks_cache = {}
    for operation in recipe["operations"]:
        if operation["kind"] in {"insert", "replace"}:
            source_definition = _definition_by_id(reviewed, operation["sourceDefinitionId"])
            source_key = (
                source_definition["pipelineClass"],
                "__unpruned__"
                if operation["sourceExecutionScope"] == "unpruned_pipeline"
                else source_definition["workflowId"],
            )
            if source_key not in source_blocks_cache:
                source_blocks_cache[source_key] = resolver(*source_key)
            source_blocks = source_blocks_cache[source_key]
            source_path = tuple(operation["sourcePath"])
            try:
                selected = _block_at_path(source_blocks, source_path)
            except ModularCompositionError:
                # The reviewed default-workflow inventory uses flattened dotted
                # keys for Edit Plus/Layered, but their native default trees stay
                # nested. This fallback applies only to an already authenticated
                # selected-workflow path, never arbitrary execution paths. Exact
                # native keys take precedence and the class is checked below.
                if operation["sourceExecutionScope"] != "selected_workflow" or not any("." in part for part in source_path):
                    raise
                selected = _block_at_path(source_blocks, tuple(part for segment in source_path for part in segment.split(".")))
            expected_source_class = definitions[operation["sourceBlockDefinitionId"]]["className"]
            if type(selected).__name__ != expected_source_class:
                raise ModularCompositionError(
                    f"Runtime source block {'.'.join(source_path)!r} is {type(selected).__name__}, "
                    f"not reviewed {expected_source_class}."
                )
            selected = deepcopy(selected)
            if operation["kind"] == "insert":
                parent_path = tuple(operation["parentPath"])
                target_parent = _block_at_path(blocks, parent_path) if parent_path else blocks
                _insert(getattr(target_parent, "sub_blocks", None), operation["name"], selected, operation["index"])
            else:
                target_path = tuple(operation["path"])
                target_parent_path = target_path[:-1]
                target_parent = _block_at_path(blocks, target_parent_path) if target_parent_path else blocks
                target_sub_blocks = getattr(target_parent, "sub_blocks", None)
                if not isinstance(target_sub_blocks, Mapping) or target_path[-1] not in target_sub_blocks:
                    raise ModularCompositionError(
                        f"Runtime composition target {'.'.join(target_path)!r} disappeared."
                    )
                target_names = list(target_sub_blocks)
                target_index = target_names.index(target_path[-1])
                del target_sub_blocks[target_path[-1]]
                _insert(target_sub_blocks, target_path[-1], selected, target_index)
            continue
        source_path = tuple(operation["path"])
        source_parent_path = source_path[:-1]
        source_parent = _block_at_path(blocks, source_parent_path) if source_parent_path else blocks
        source_sub_blocks = getattr(source_parent, "sub_blocks", None)
        if not isinstance(source_sub_blocks, Mapping) or source_path[-1] not in source_sub_blocks:
            raise ModularCompositionError(f"Runtime composition source {'.'.join(source_path)!r} disappeared.")
        if operation["kind"] == "remove":
            del source_sub_blocks[source_path[-1]]
            continue
        selected = source_sub_blocks[source_path[-1]]
        if operation["kind"] == "duplicate":
            selected = deepcopy(selected)
        else:
            del source_sub_blocks[source_path[-1]]
        parent_path = tuple(operation["parentPath"])
        target_parent = _block_at_path(blocks, parent_path) if parent_path else blocks
        target_sub_blocks = getattr(target_parent, "sub_blocks", None)
        _insert(target_sub_blocks, operation["name"], selected, operation["index"])

    return recipe, blocks


def rebuild_reviewed_modular_composition(
    value: Any,
    *,
    library: Mapping[str, Any] | None = None,
    blocks_resolver: Callable[[str, str], Any] | None = None,
) -> dict[str, Any]:
    """Rebuild a structural preview; this receipt never grants runtime admission."""
    recipe, blocks = build_reviewed_modular_composition_blocks(
        value, library=library, blocks_resolver=blocks_resolver,
    )
    # Inspect the edited tree, not the original workflow's branch selection.
    # A concrete child may have moved out of a conditional container. Its old
    # selector is not the executor of the explicit MoDiff graph, and calling it
    # here can fail even though each graph-selected step initializes and runs.
    selected_blocks = blocks
    init_pipeline = getattr(selected_blocks, "init_pipeline", None)
    if not callable(init_pipeline):
        raise ModularCompositionError("The composed upstream block tree cannot initialize a Modular pipeline.")
    pipeline = init_pipeline()
    component_names = sorted(str(name) for name in getattr(pipeline, "pretrained_component_names", ()) if name)
    body = {
        "schemaVersion": MODULAR_COMPOSITION_SCHEMA_VERSION,
        "claim": "reviewed_modular_composition_rebuilt",
        "executable": False,
        "inspectionScope": "edited_unpruned_tree",
        "recipeHash": recipe["recipeHash"],
        "diffusersRevision": recipe["diffusersRevision"],
        "pipelineClass": recipe["pipelineClass"],
        "workflowId": recipe["workflowId"],
        "definitionId": recipe["definitionId"],
        "blockContractHash": recipe["blockContractHash"],
        "inputs": _field_names(getattr(selected_blocks, "inputs", ())),
        "outputs": _field_names(getattr(selected_blocks, "outputs", ())),
        "components": component_names,
        "configs": _field_names(getattr(selected_blocks, "expected_configs", ())),
        "composedPaths": recipe["composedPaths"],
    }
    return {**body, "receiptHash": _hash(body)}
