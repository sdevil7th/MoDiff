"""Pinned, no-weight contracts for individual Modular Diffusers blocks.

The existing workflow snapshot intentionally remains stable.  This companion
snapshot records the exact block contracts and unambiguous ``sub_blocks``
membership needed by the visual graph compiler.  Runtime readers only parse a
reviewed JSON artifact; importing this module never imports Diffusers, Torch,
Transformers, or a model stack.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, get_args, get_origin

from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION
from modiff.modular_workflow_discovery import load_reviewed_modular_workflow_snapshot


MODULAR_BLOCK_CONTRACT_SCHEMA_VERSION = 1
MODULAR_BLOCK_CONTRACT_SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / "modular-block-contracts.json"

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,255}$")
_BLOCK_DEFINITION_ID = re.compile(
    r"^diffusers\.modular-block:[A-Za-z_][A-Za-z0-9_]{0,255}:sha256:[0-9a-f]{64}$"
)
_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_SNAPSHOT_BYTES = 32 * 1024 * 1024
_MAX_BLOCK_DEFINITIONS = 2048
_MAX_WORKFLOWS = 256
_MAX_PLACEMENTS = 2048
_MAX_PATH_DEPTH = 64
_MAX_FIELDS = 512
_MAX_COMPONENTS = 256
_MAX_CONFIGS = 256
_MAX_JSON_COLLECTION = 256
_MAX_JSON_DEPTH = 8


class ModularBlockContractError(ValueError):
    """A Modular Diffusers block contract is malformed or ambiguous."""


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ModularBlockContractError(f"{label} {value!r} must be a bounded identifier.")
    return value


def _description(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:1024]


def _block_description(block: Any) -> str:
    try:
        return _description(block.description)
    except (AttributeError, NotImplementedError):
        return ""


def _content_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


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


def _json_value(value: Any, *, strict: bool, depth: int = 0) -> Any:
    if depth > _MAX_JSON_DEPTH:
        if strict:
            raise ModularBlockContractError("Block contract JSON value exceeds the nesting limit.")
        return None
    if value is None or type(value) in {bool, str}:
        return value
    if type(value) in {int, float}:
        if isinstance(value, float) and not math.isfinite(value):
            if strict:
                raise ModularBlockContractError("Block contract JSON numbers must be finite.")
            return None
        return value
    if isinstance(value, Mapping):
        if len(value) > _MAX_JSON_COLLECTION or not all(isinstance(key, str) and len(key) <= 256 for key in value):
            if strict:
                raise ModularBlockContractError("Block contract JSON mapping is oversized or malformed.")
            return None
        normalized = {
            key: _json_value(item, strict=strict, depth=depth + 1)
            for key, item in value.items()
        }
        if not strict and any(item is None and value[key] is not None for key, item in normalized.items()):
            return None
        return normalized
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_JSON_COLLECTION:
            if strict:
                raise ModularBlockContractError("Block contract JSON sequence is oversized.")
            return None
        normalized = [_json_value(item, strict=strict, depth=depth + 1) for item in value]
        if not strict and any(item is None and source is not None for item, source in zip(normalized, value)):
            return None
        return normalized
    if strict:
        raise ModularBlockContractError(f"Unsupported block contract JSON value {type(value).__name__}.")
    return None


def _block_kind(block: Any) -> str:
    names = {base.__name__ for base in type(block).__mro__}
    if "LoopSequentialPipelineBlocks" in names:
        return "loop"
    if "AutoPipelineBlocks" in names or getattr(block, "_workflow_map", None):
        return "auto"
    if "ConditionalPipelineBlocks" in names:
        return "conditional"
    if "SequentialPipelineBlocks" in names:
        return "sequential"
    return "block"


def _field_contract(field: Any, *, required_names: set[str]) -> dict[str, Any]:
    name = _name(getattr(field, "name", None), "Block field name")
    kwargs_type = getattr(field, "kwargs_type", None)
    return {
        "name": name,
        "type": _type_name(getattr(field, "type_hint", None)),
        "required": name in required_names or getattr(field, "required", False) is True,
        "default": _json_value(getattr(field, "default", None), strict=False),
        "description": _description(getattr(field, "description", "")),
        "kwargsType": _name(kwargs_type, "Block field kwargs type") if kwargs_type is not None else None,
    }


def _variadic_input_contract(field: Any) -> dict[str, Any]:
    kwargs_type = _name(getattr(field, "kwargs_type", None), "Variadic block input kwargs type")
    return {
        "kwargsType": kwargs_type,
        "type": _type_name(getattr(field, "type_hint", None)),
        "required": getattr(field, "required", False) is True,
        "default": _json_value(getattr(field, "default", None), strict=False),
        "description": _description(getattr(field, "description", "")),
    }


def _component_contract(component: Any) -> dict[str, Any]:
    config = getattr(component, "config", None)
    return {
        "name": _name(getattr(component, "name", None), "Block component name"),
        "type": _type_name(getattr(component, "type_hint", None)),
        "description": _description(getattr(component, "description", "")),
        "creationMethod": str(getattr(component, "default_creation_method", ""))[:64],
        "defaultConfig": _json_value(config, strict=True) if config is not None else None,
    }


def _config_contract(config: Any) -> dict[str, Any]:
    return {
        "name": _name(getattr(config, "name", None), "Block config name"),
        "default": _json_value(getattr(config, "default", None), strict=True),
        "description": _description(getattr(config, "description", "")),
    }


def _block_definition(block: Any) -> dict[str, Any]:
    input_fields = list(getattr(block, "inputs", ()))
    # Upstream commonly exposes ``required_inputs`` as a set. Canonicalize it
    # so the reviewed artifact and content-derived identities do not vary with
    # Python hash randomization between otherwise identical processes.
    required_inputs = sorted(
        _name(name, "Required block input") for name in getattr(block, "required_inputs", ())
    )
    named_inputs = [field for field in input_fields if isinstance(getattr(field, "name", None), str)]
    variadic_inputs = [field for field in input_fields if getattr(field, "name", None) is None]
    output_fields = [
        field for field in getattr(block, "outputs", ()) if isinstance(getattr(field, "name", None), str)
    ]
    body = {
        "schemaVersion": MODULAR_BLOCK_CONTRACT_SCHEMA_VERSION,
        "className": _name(type(block).__name__, "Block class"),
        "kind": _block_kind(block),
        "description": _block_description(block),
        "inputs": [_field_contract(field, required_names=set(required_inputs)) for field in named_inputs],
        "variadicInputs": [_variadic_input_contract(field) for field in variadic_inputs],
        "requiredInputs": required_inputs,
        "outputs": [_field_contract(field, required_names=set()) for field in output_fields],
        "components": [_component_contract(component) for component in getattr(block, "expected_components", ())],
        "configs": [_config_contract(config) for config in getattr(block, "expected_configs", ())],
    }
    content_hash = _content_hash(body)
    return {
        **body,
        "id": f"diffusers.modular-block:{body['className']}:{content_hash}",
        "contentHash": content_hash,
    }


def build_modular_block_contracts(pipeline: Any) -> dict[str, Any]:
    """Discover exact block definitions and placements from one no-weight pipeline."""

    pipeline_class = _name(type(pipeline).__name__, "Pipeline class")
    blocks = pipeline.blocks
    raw_map = getattr(blocks, "_workflow_map", None)
    workflow_map = raw_map if isinstance(raw_map, Mapping) and raw_map else None
    if workflow_map is not None:
        workflow_ids = list(blocks.available_workflows)
        if set(workflow_ids) != set(workflow_map):
            raise ModularBlockContractError("available_workflows disagrees with the upstream workflow map.")
    else:
        try:
            advertised = list(blocks.available_workflows)
        except NotImplementedError:
            advertised = []
        if advertised:
            raise ModularBlockContractError("A non-Auto block unexpectedly advertises named workflows.")
        workflow_ids = ["default"]

    definitions: dict[str, dict[str, Any]] = {}
    workflows = []
    for workflow_id_value in workflow_ids:
        workflow_id = _name(workflow_id_value, "Workflow id")
        selected = blocks.get_workflow(workflow_id) if workflow_map is not None else blocks
        workflow = selected.get_execution_blocks()
        root_definition = _block_definition(workflow)
        previous_root_definition = definitions.get(root_definition["id"])
        if previous_root_definition is not None and previous_root_definition != root_definition:
            raise ModularBlockContractError("A root block definition id resolved to inconsistent contracts.")
        definitions[root_definition["id"]] = root_definition
        placements: list[dict[str, Any]] = []

        def visit(block: Any, path: tuple[str, ...], order: int) -> None:
            if len(placements) >= _MAX_PLACEMENTS:
                raise ModularBlockContractError("Workflow exceeds the block placement limit.")
            definition = _block_definition(block)
            previous = definitions.get(definition["id"])
            if previous is not None and previous != definition:
                raise ModularBlockContractError("A block definition id resolved to inconsistent contracts.")
            definitions[definition["id"]] = definition
            placements.append(
                {
                    "path": list(path),
                    "legacyPath": ".".join(path),
                    "order": order,
                    "blockDefinitionId": definition["id"],
                }
            )
            nested = getattr(block, "sub_blocks", None)
            if not isinstance(nested, Mapping):
                return
            for child_order, (child_name_value, child) in enumerate(nested.items()):
                child_name = _name(child_name_value, "Sub-block name")
                visit(child, (*path, child_name), child_order)

        nested = getattr(workflow, "sub_blocks", None)
        if not isinstance(nested, Mapping):
            raise ModularBlockContractError("Upstream workflow blocks do not expose a sub-block mapping.")
        for order, (block_name_value, block) in enumerate(nested.items()):
            block_name = _name(block_name_value, "Top-level block name")
            visit(block, (block_name,), order)
        body = {
            "schemaVersion": MODULAR_BLOCK_CONTRACT_SCHEMA_VERSION,
            "pipelineClass": pipeline_class,
            "workflowId": workflow_id,
            "workflowKind": _block_kind(workflow),
            "rootBlockDefinitionId": root_definition["id"],
            "placements": placements,
        }
        workflows.append({**body, "contentHash": _content_hash(body)})
    return {
        "blockDefinitions": sorted(definitions.values(), key=lambda item: item["id"]),
        "workflows": sorted(workflows, key=lambda item: (item["pipelineClass"], item["workflowId"])),
    }


def merge_modular_block_contracts(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    definitions: dict[str, dict[str, Any]] = {}
    workflows: list[dict[str, Any]] = []
    for part in parts:
        for definition in part.get("blockDefinitions", ()):
            previous = definitions.get(definition["id"])
            if previous is not None and previous != definition:
                raise ModularBlockContractError("A shared block id resolved to inconsistent definitions.")
            definitions[definition["id"]] = definition
        workflows.extend(part.get("workflows", ()))
    return validate_modular_block_snapshot(
        {
            "schemaVersion": MODULAR_BLOCK_CONTRACT_SCHEMA_VERSION,
            "diffusersRevision": PINNED_DIFFUSERS_REVISION,
            "blockDefinitions": sorted(definitions.values(), key=lambda item: item["id"]),
            "workflows": sorted(workflows, key=lambda item: (item["pipelineClass"], item["workflowId"])),
        }
    )


def _validate_field(value: Any, *, label: str) -> dict[str, Any]:
    keys = {"name", "type", "required", "default", "description", "kwargsType"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ModularBlockContractError(f"{label} is malformed.")
    _name(value.get("name"), f"{label} name")
    if not isinstance(value.get("type"), str) or not value["type"] or len(value["type"]) > 256:
        raise ModularBlockContractError(f"{label} type is malformed.")
    if not isinstance(value.get("required"), bool):
        raise ModularBlockContractError(f"{label} required marker is malformed.")
    if not isinstance(value.get("description"), str) or len(value["description"]) > 1024:
        raise ModularBlockContractError(f"{label} description is malformed.")
    kwargs_type = value.get("kwargsType")
    if kwargs_type is not None:
        _name(kwargs_type, f"{label} kwargs type")
    _json_value(value.get("default"), strict=True)
    return value


def _validate_block_definition(value: Any) -> dict[str, Any]:
    keys = {
        "schemaVersion",
        "id",
        "className",
        "kind",
        "description",
        "inputs",
        "variadicInputs",
        "requiredInputs",
        "outputs",
        "components",
        "configs",
        "contentHash",
    }
    if not isinstance(value, dict) or set(value) != keys or value.get("schemaVersion") != 1:
        raise ModularBlockContractError("Block definition is malformed.")
    class_name = _name(value.get("className"), "Block class")
    if value.get("kind") not in {"auto", "conditional", "sequential", "loop", "block"}:
        raise ModularBlockContractError("Block kind is malformed.")
    if not isinstance(value.get("description"), str) or len(value["description"]) > 1024:
        raise ModularBlockContractError("Block description is malformed.")
    for collection in ("inputs", "variadicInputs", "requiredInputs", "outputs", "components", "configs"):
        if not isinstance(value.get(collection), list):
            raise ModularBlockContractError(f"Block {collection} collection is malformed.")
    if len(value["inputs"]) > _MAX_FIELDS or len(value["outputs"]) > _MAX_FIELDS:
        raise ModularBlockContractError("Block field collection exceeds its limit.")
    for field in value["inputs"]:
        _validate_field(field, label="Block input")
    for field in value["outputs"]:
        _validate_field(field, label="Block output")
    required_inputs = [_name(name, "Required block input") for name in value["requiredInputs"]]
    if len(required_inputs) != len(set(required_inputs)):
        raise ModularBlockContractError("Required block inputs must be unique.")
    if len(value["variadicInputs"]) > _MAX_FIELDS:
        raise ModularBlockContractError("Block variadic input collection exceeds its limit.")
    variadic_types = []
    for item in value["variadicInputs"]:
        if not isinstance(item, dict) or set(item) != {"kwargsType", "type", "required", "default", "description"}:
            raise ModularBlockContractError("Variadic block input is malformed.")
        variadic_types.append(_name(item.get("kwargsType"), "Variadic block input kwargs type"))
        if not isinstance(item.get("type"), str) or not item["type"] or len(item["type"]) > 256:
            raise ModularBlockContractError("Variadic block input type is malformed.")
        if not isinstance(item.get("required"), bool):
            raise ModularBlockContractError("Variadic block input required marker is malformed.")
        if not isinstance(item.get("description"), str) or len(item["description"]) > 1024:
            raise ModularBlockContractError("Variadic block input description is malformed.")
        _json_value(item.get("default"), strict=True)
    if len(variadic_types) != len(set(variadic_types)):
        raise ModularBlockContractError("Variadic block input kwargs types must be unique.")
    if len(value["components"]) > _MAX_COMPONENTS or len(value["configs"]) > _MAX_CONFIGS:
        raise ModularBlockContractError("Block component or config collection exceeds its limit.")
    component_names = []
    for component in value["components"]:
        if not isinstance(component, dict) or set(component) != {
            "name",
            "type",
            "description",
            "creationMethod",
            "defaultConfig",
        }:
            raise ModularBlockContractError("Block component is malformed.")
        component_names.append(_name(component.get("name"), "Block component name"))
        if not all(isinstance(component.get(key), str) for key in ("type", "description", "creationMethod")):
            raise ModularBlockContractError("Block component text metadata is malformed.")
        if not component["type"] or len(component["type"]) > 256 or len(component["description"]) > 1024:
            raise ModularBlockContractError("Block component metadata exceeds its limit.")
        _json_value(component.get("defaultConfig"), strict=True)
    if len(component_names) != len(set(component_names)):
        raise ModularBlockContractError("Block component names must be unique.")
    config_names = []
    for config in value["configs"]:
        if not isinstance(config, dict) or set(config) != {"name", "default", "description"}:
            raise ModularBlockContractError("Block config is malformed.")
        config_names.append(_name(config.get("name"), "Block config name"))
        if not isinstance(config.get("description"), str) or len(config["description"]) > 1024:
            raise ModularBlockContractError("Block config description is malformed.")
        _json_value(config.get("default"), strict=True)
    if len(config_names) != len(set(config_names)):
        raise ModularBlockContractError("Block config names must be unique.")
    body = {key: item for key, item in value.items() if key not in {"id", "contentHash"}}
    content_hash = value.get("contentHash")
    expected_hash = _content_hash(body)
    expected_id = f"diffusers.modular-block:{class_name}:{expected_hash}"
    if (
        not isinstance(content_hash, str)
        or _CONTENT_HASH.fullmatch(content_hash) is None
        or content_hash != expected_hash
        or value.get("id") != expected_id
        or _BLOCK_DEFINITION_ID.fullmatch(expected_id) is None
    ):
        raise ModularBlockContractError("Block definition identity or content hash is invalid.")
    return value


def validate_modular_block_snapshot(
    value: Any,
    *,
    workflow_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate, cross-check, and detach one complete block snapshot."""

    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        normalized = json.loads(encoded)
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ModularBlockContractError(f"Block snapshot is not finite JSON: {error}") from error
    if len(encoded.encode("utf-8")) > _MAX_SNAPSHOT_BYTES:
        raise ModularBlockContractError("Block snapshot exceeds the 32 MiB limit.")
    if not isinstance(normalized, dict) or set(normalized) != {
        "schemaVersion",
        "diffusersRevision",
        "blockDefinitions",
        "workflows",
    }:
        raise ModularBlockContractError("Block snapshot has missing or unknown top-level fields.")
    if (
        normalized.get("schemaVersion") != MODULAR_BLOCK_CONTRACT_SCHEMA_VERSION
        or normalized.get("diffusersRevision") != PINNED_DIFFUSERS_REVISION
    ):
        raise ModularBlockContractError("Block snapshot revision is not the pinned Diffusers revision.")
    definitions = normalized.get("blockDefinitions")
    workflows = normalized.get("workflows")
    if not isinstance(definitions, list) or not 0 < len(definitions) <= _MAX_BLOCK_DEFINITIONS:
        raise ModularBlockContractError("Block definition collection is malformed.")
    if not isinstance(workflows, list) or not 0 < len(workflows) <= _MAX_WORKFLOWS:
        raise ModularBlockContractError("Block workflow collection is malformed.")
    definitions_by_id: dict[str, dict[str, Any]] = {}
    for definition in definitions:
        validated = _validate_block_definition(definition)
        if validated["id"] in definitions_by_id:
            raise ModularBlockContractError("Block definition ids must be unique.")
        definitions_by_id[validated["id"]] = validated

    workflows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    container_definition_ids: set[str] = set()
    referenced_definition_ids: set[str] = set()
    for workflow in workflows:
        if not isinstance(workflow, dict) or set(workflow) != {
            "schemaVersion",
            "pipelineClass",
            "workflowId",
            "workflowKind",
            "rootBlockDefinitionId",
            "placements",
            "contentHash",
        }:
            raise ModularBlockContractError("Block workflow is malformed.")
        if workflow.get("schemaVersion") != MODULAR_BLOCK_CONTRACT_SCHEMA_VERSION:
            raise ModularBlockContractError("Block workflow schema version is invalid.")
        key = (
            _name(workflow.get("pipelineClass"), "Block workflow pipeline class"),
            _name(workflow.get("workflowId"), "Block workflow id"),
        )
        if key in workflows_by_key:
            raise ModularBlockContractError("Block workflow identities must be unique.")
        if workflow.get("workflowKind") not in {"auto", "sequential", "loop", "block"}:
            raise ModularBlockContractError("Block workflow kind is malformed.")
        root_definition_id = workflow.get("rootBlockDefinitionId")
        if root_definition_id not in definitions_by_id:
            raise ModularBlockContractError("Block workflow references an unknown root definition.")
        if definitions_by_id[root_definition_id]["kind"] != workflow["workflowKind"]:
            raise ModularBlockContractError("Block workflow root kind is inconsistent.")
        container_definition_ids.add(root_definition_id)
        referenced_definition_ids.add(root_definition_id)
        placements = workflow.get("placements")
        if not isinstance(placements, list) or not placements or len(placements) > _MAX_PLACEMENTS:
            raise ModularBlockContractError("Block workflow placements are malformed.")
        paths: set[tuple[str, ...]] = set()
        definition_id_by_path: dict[tuple[str, ...], str] = {}
        orders_by_parent: dict[tuple[str, ...], list[int]] = {}
        for placement in placements:
            if not isinstance(placement, dict) or set(placement) != {
                "path",
                "legacyPath",
                "order",
                "blockDefinitionId",
            }:
                raise ModularBlockContractError("Block placement is malformed.")
            raw_path = placement.get("path")
            if not isinstance(raw_path, list) or not 0 < len(raw_path) <= _MAX_PATH_DEPTH:
                raise ModularBlockContractError("Block placement path is malformed.")
            path = tuple(_name(segment, "Block placement path segment") for segment in raw_path)
            if path in paths or placement.get("legacyPath") != ".".join(path):
                raise ModularBlockContractError("Block placement path is duplicated or inconsistent.")
            paths.add(path)
            order = placement.get("order")
            if not isinstance(order, int) or isinstance(order, bool) or order < 0:
                raise ModularBlockContractError("Block placement order is malformed.")
            orders_by_parent.setdefault(path[:-1], []).append(order)
            if placement.get("blockDefinitionId") not in definitions_by_id:
                raise ModularBlockContractError("Block placement references an unknown definition.")
            definition_id_by_path[path] = placement["blockDefinitionId"]
            referenced_definition_ids.add(placement["blockDefinitionId"])
        for path in paths:
            if len(path) > 1 and path[:-1] not in paths:
                raise ModularBlockContractError("Block placement references a missing explicit parent.")
            if any(candidate[:-1] == path for candidate in paths):
                container_definition_ids.add(definition_id_by_path[path])
        if any(sorted(orders) != list(range(len(orders))) for orders in orders_by_parent.values()):
            raise ModularBlockContractError("Sibling block placement order must be contiguous and unique.")
        body = {item: content for item, content in workflow.items() if item != "contentHash"}
        if workflow.get("contentHash") != _content_hash(body):
            raise ModularBlockContractError("Block workflow content hash is invalid.")
        workflows_by_key[key] = workflow

    reviewed = workflow_snapshot or load_reviewed_modular_workflow_snapshot()
    expected_workflows = {
        (contract["pipelineClass"], workflow["id"]): workflow
        for contract in reviewed["contracts"]
        for workflow in contract["workflows"]
    }
    if set(workflows_by_key) != set(expected_workflows):
        raise ModularBlockContractError("Block workflows do not exactly cover the reviewed workflow snapshot.")
    for key, workflow in workflows_by_key.items():
        expected = expected_workflows[key]
        if workflow["workflowKind"] != expected["kind"]:
            raise ModularBlockContractError("Block workflow kind disagrees with the reviewed workflow snapshot.")
        placements_by_legacy_path = {placement["legacyPath"]: placement for placement in workflow["placements"]}
        expected_steps = {step["path"]: step for step in expected["steps"]}
        if set(placements_by_legacy_path) != set(expected_steps):
            raise ModularBlockContractError("Block placements disagree with the reviewed workflow steps.")
        for legacy_path, placement in placements_by_legacy_path.items():
            definition = definitions_by_id[placement["blockDefinitionId"]]
            step = expected_steps[legacy_path]
            if (
                definition["className"] != step["className"]
                or definition["kind"] != step["kind"]
                or definition["description"] != step["description"]
            ):
                raise ModularBlockContractError("Block definition disagrees with its reviewed workflow step.")
    if referenced_definition_ids != set(definitions_by_id):
        raise ModularBlockContractError("Block snapshot contains unreferenced definitions.")
    for definition_id, definition in definitions_by_id.items():
        if definition_id in container_definition_ids:
            continue
        input_names = {field["name"] for field in definition["inputs"]}
        if not set(definition["requiredInputs"]).issubset(input_names):
            raise ModularBlockContractError("Leaf block required inputs must be declared inputs.")
    return normalized


@lru_cache(maxsize=1)
def load_reviewed_modular_block_snapshot(path: Path = MODULAR_BLOCK_CONTRACT_SNAPSHOT) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ModularBlockContractError(f"Cannot load reviewed block snapshot: {error}") from error
    return validate_modular_block_snapshot(value)


def reviewed_modular_block_snapshot() -> dict[str, Any]:
    return json.loads(json.dumps(load_reviewed_modular_block_snapshot(), allow_nan=False))
