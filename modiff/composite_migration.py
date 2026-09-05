"""Preview, apply, and recover the legacy-composite to Block V2 migration.

The default operation is a deterministic, read-only preview. Applying a plan
requires an exact preview identifier plus a literal operator confirmation.
Every changed source is backed up byte-for-byte before the first replacement;
rollback refuses to overwrite a file whose bytes match neither the reviewed
source nor the applied result.

V1 User Node definitions and workflow instances have a complete additive
reader and are convertible. Legacy registered Clusters remain blocked unless
the caller supplies an exact, source-bound compiler supplement produced from a
backend-pinned registered ``BlockDefinitionV2``.  The backend validates that
output and its preservation receipts; it never infers an execution definition
from presentation children.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from modiff.block_definition_v2 import (
    block_definition_canonical_sha256_v2,
    block_definition_content_hash_v2,
    block_graph_hash_v2,
    block_interface_hash_v2,
    validate_block_definition_v2,
    validate_block_instance_v2,
)
from modiff.composite_migration_inventory import _build_from_sources, _load_sources
from modiff.studio_persistence_lock import STUDIO_PERSISTENCE_LOCK


MIGRATION_SCHEMA_VERSION = 1
COMPILER_SUPPLEMENT_SCHEMA_VERSION = 1
APPLY_CONFIRMATION = "APPLY_BLOCK_V2_MIGRATION"
ROLLBACK_CONFIRMATION = "ROLLBACK_BLOCK_V2_MIGRATION"
_MIGRATION_ID = re.compile(r"^block-v2-migration-[0-9a-f]{24}$")
_SAFE_DEFINITION_FILE_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_PREVIEW_MEDIA = {
    "ui_image": "image",
    "ui_video": "video",
    "ui_audio": "audio",
    "ui_text": "text",
    "ui_imagecompare": "image",
}
_LEGACY_PRESENTATION_DATA = {
    "uiState",
    "executionProgress",
    "outputSummary",
    "userBlockInstanceId",
    "userBlockSourceNodeId",
}
_PARAM_PRESENTATION_FIELDS = {"label", "type", "display", "isConnected", "fieldOptions"}
_LEGACY_CHILD_LEFT = 28
_LEGACY_CHILD_TOP = 76
_COMPILER_SUPPLEMENT_KIND = "registered_cluster_v2_compiler_supplement"
_CATALOG_SOURCE_KINDS = {"diffusers_catalog", "transformers_catalog"}


class CompositeMigrationError(ValueError):
    """Base error safe to return as a bounded operator-facing response."""


class CompositeMigrationAuthorizationError(CompositeMigrationError):
    """The caller did not supply exact explicit mutation authority."""


class CompositeMigrationConflictError(CompositeMigrationError):
    """Source or recovery bytes no longer match the reviewed transaction."""


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _array(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _encoded_document(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _legacy_record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CompositeMigrationError(f"{label} must be a non-empty string.")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    label: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    keys = set(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise CompositeMigrationError(f"{label} is missing field(s): {', '.join(sorted(missing))}.")
    if unknown:
        raise CompositeMigrationError(f"{label} contains unknown field(s): {', '.join(sorted(unknown))}.")


def _registered_definition_pins() -> Mapping[str, tuple[str, str]]:
    # Imported lazily so read-only migration inventory does not initialize the
    # optional-runtime qualification module unless a compiler output is used.
    from modiff.huggingface_cluster_runtime import REGISTERED_BLOCK_V2_DEFINITION_PINS

    return REGISTERED_BLOCK_V2_DEFINITION_PINS


def _semantic_equivalence_receipt_for_historical(**identity: str) -> dict[str, Any] | None:
    # Imported lazily so the default V1 inventory remains independent of the
    # reviewed historical-equivalence ledger.
    from modiff.legacy_cluster_semantic_equivalence import (
        semantic_equivalence_receipt_for_historical,
    )

    return semantic_equivalence_receipt_for_historical(**identity)


def _semantic_equivalence_receipt_by_reference(
    receipt_id: str, receipt_hash: str
) -> dict[str, Any] | None:
    from modiff.legacy_cluster_semantic_equivalence import (
        semantic_equivalence_receipt_by_reference,
    )

    return semantic_equivalence_receipt_by_reference(receipt_id, receipt_hash)


def _historical_compiler_mapping_for_historical(**identity: str) -> dict[str, Any] | None:
    from modiff.legacy_cluster_compiler_mappings import (
        historical_compiler_mapping_for_historical,
    )

    return historical_compiler_mapping_for_historical(**identity)


def _historical_compiler_mapping_by_reference(
    mapping_id: str, mapping_hash: str
) -> dict[str, Any] | None:
    from modiff.legacy_cluster_compiler_mappings import (
        historical_compiler_mapping_by_reference,
    )

    return historical_compiler_mapping_by_reference(mapping_id, mapping_hash)


def _semantic_equivalence_authority(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete non-instance review diff safe for preview/journal UI."""

    return {
        "receiptId": receipt.get("id"),
        "receiptHash": receipt.get("receiptHash"),
        "historical": copy.deepcopy(_object(receipt.get("historical"))),
        "destination": copy.deepcopy(_object(receipt.get("destination"))),
        "review": copy.deepcopy(_object(receipt.get("review"))),
    }


def _historical_compiler_mapping_authority(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Return the reviewed archived-body compiler destination for preview."""

    return {
        "mappingId": mapping.get("id"),
        "mappingHash": mapping.get("mappingHash"),
        "historical": copy.deepcopy(_object(mapping.get("historical"))),
        "destination": copy.deepcopy(_object(mapping.get("destination"))),
        "review": copy.deepcopy(_object(mapping.get("review"))),
    }


def _legacy_cluster_identity(root: Mapping[str, Any]) -> dict[str, Any]:
    instance = _object(_object(root.get("data")).get("huggingFaceClusterInstance"))
    definition = _object(instance.get("definition"))
    execution = _object(instance.get("execution"))
    return {
        "manifestDefinitionId": _text(definition.get("id")),
        "libraryRevision": _text(definition.get("libraryRevision")),
        "manifestContentHash": _text(definition.get("contentHash")),
        "executionAdmissionId": _text(execution.get("admissionId")),
        "studioExecutionSpec": copy.deepcopy(_object(execution.get("studioExecutionSpec"))),
    }


def _available_semantic_equivalence_authority(root: Mapping[str, Any]) -> dict[str, Any] | None:
    identity = _legacy_cluster_identity(root)
    required = (
        "manifestDefinitionId",
        "libraryRevision",
        "manifestContentHash",
        "executionAdmissionId",
    )
    if any(not identity.get(field) for field in required):
        return None
    receipt = _semantic_equivalence_receipt_for_historical(
        manifest_definition_id=identity["manifestDefinitionId"],
        library_revision=identity["libraryRevision"],
        manifest_content_hash=identity["manifestContentHash"],
        execution_admission_id=identity["executionAdmissionId"],
    )
    if receipt is None:
        return None
    historical = _object(receipt.get("historical"))
    # The ledger lookup uses the immutable manifest/admission tuple, but the
    # visible preview must also bind the exact Studio execution contract.  Do
    # not advertise a review receipt whose execution spec differs from the
    # saved historical root; apply already repeats this check fail-closed.
    for field in (*required, "studioExecutionSpec"):
        if historical.get(field) != identity.get(field):
            return None
    return _semantic_equivalence_authority(receipt)


def _available_historical_compiler_mapping_authority(
    root: Mapping[str, Any],
) -> dict[str, Any] | None:
    identity = _legacy_cluster_identity(root)
    required = (
        "manifestDefinitionId",
        "libraryRevision",
        "manifestContentHash",
        "executionAdmissionId",
    )
    if any(not identity.get(field) for field in required):
        return None
    mapping = _historical_compiler_mapping_for_historical(
        manifest_definition_id=identity["manifestDefinitionId"],
        library_revision=identity["libraryRevision"],
        manifest_content_hash=identity["manifestContentHash"],
        execution_admission_id=identity["executionAdmissionId"],
    )
    if mapping is None:
        return None
    historical = _object(mapping.get("historical"))
    for field in (*required, "studioExecutionSpec"):
        if historical.get(field) != identity.get(field):
            return None
    return _historical_compiler_mapping_authority(mapping)


def _strip_legacy_node_data(node: Mapping[str, Any]) -> dict[str, Any]:
    data = _object(node.get("data"))
    return copy.deepcopy({key: value for key, value in data.items() if key not in _LEGACY_PRESENTATION_DATA})


def _value_type_from_legacy(value: Any) -> str:
    if isinstance(value, list):
        return "|".join(str(item) for item in value) if value else "any"
    return value if isinstance(value, str) and value else "any"


def _port_from_legacy(value: Any, label: str) -> dict[str, Any]:
    port = _object(value)
    return {
        "portId": _required_text(port.get("id"), f"{label}.id"),
        "label": _required_text(port.get("label"), f"{label}.label"),
        "valueType": _value_type_from_legacy(port.get("type")),
        "required": False,
        "binding": {
            "nodeId": _required_text(port.get("nodeId"), f"{label}.nodeId"),
            "fieldOrPortId": _required_text(port.get("paramKey"), f"{label}.paramKey"),
        },
    }


def _legacy_param(block: Mapping[str, Any], node_id: str, field_id: str) -> dict[str, Any]:
    for candidate in _array(block.get("nodes")):
        node = _object(candidate)
        if node.get("id") == node_id:
            return _object(_object(_object(node.get("data")).get("params")).get(field_id))
    return {}


def _graph_from_v1(block: Mapping[str, Any], controls: Sequence[dict[str, Any]]) -> dict[str, Any]:
    nodes = []
    for index, value in enumerate(_array(block.get("nodes"))):
        candidate = _object(value)
        data = _object(candidate.get("data"))
        nodes.append(
            {
                "nodeId": _required_text(candidate.get("id"), f"legacy User Node nodes[{index}].id"),
                "nodeType": (_text(candidate.get("type")) or _text(data.get("type")) or "custom"),
                "data": _strip_legacy_node_data(candidate),
            }
        )
    studio_controls = [control for control in controls if control.get("kind") == "studio-form"]
    studio_binding_node_id = "__legacy_studio_form__"
    node_ids = {node["nodeId"] for node in nodes}
    while studio_binding_node_id in node_ids:
        studio_binding_node_id = "_" + studio_binding_node_id
    if studio_controls:
        nodes.append(
            {
                "nodeId": studio_binding_node_id,
                "nodeType": "modiff.legacy.studio-form-binding",
                "semanticRole": "legacy_studio_form_binding",
                "data": {
                    "hidden": True,
                    "formKeys": [control.get("formKey") or control.get("id") for control in studio_controls],
                },
            }
        )

    edges = []
    for index, value in enumerate(_array(block.get("edges"))):
        edge = _object(value)
        edges.append(
            {
                "edgeId": _required_text(edge.get("id"), f"legacy User Node edges[{index}].id"),
                "sourceNodeId": _required_text(edge.get("source"), f"legacy User Node edges[{index}].source"),
                "sourcePortId": _text(edge.get("sourceHandle")) or f"legacy-output-{index}",
                "targetNodeId": _required_text(edge.get("target"), f"legacy User Node edges[{index}].target"),
                "targetPortId": _text(edge.get("targetHandle")) or f"legacy-input-{index}",
            }
        )
    graph = {"nodes": nodes, "edges": edges}
    graph["graphHash"] = block_graph_hash_v2(graph)
    return graph


def _controls_from_v1(
    block: Mapping[str, Any], graph: Mapping[str, Any], controls: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    studio_binding_node_id = next(
        (
            node.get("nodeId")
            for node in _array(graph.get("nodes"))
            if _object(node).get("semanticRole") == "legacy_studio_form_binding"
        ),
        None,
    )
    result = []
    for order, control in enumerate(controls):
        control_id = _required_text(control.get("id"), f"legacy User Node exposedParams[{order}].id")
        graph_param = control.get("kind") == "graph-param"
        if graph_param:
            node_id = _required_text(control.get("nodeId"), f"legacy User Node control {control_id}.nodeId")
            field_id = _required_text(control.get("paramKey"), f"legacy User Node control {control_id}.paramKey")
            param = _legacy_param(block, node_id, field_id)
        else:
            node_id = _required_text(
                studio_binding_node_id,
                f"legacy User Node control {control_id} Studio binding node",
            )
            field_id = _required_text(control.get("formKey"), f"legacy User Node control {control_id}.formKey")
            param = {}
        default_present = "value" in param or "default" in param
        default_value = param.get("value") if "value" in param else param.get("default")
        migrated = {
            "controlId": control_id,
            "label": _required_text(control.get("label"), f"legacy User Node control {control_id}.label"),
            "binding": {"nodeId": node_id, "fieldId": field_id},
            "valueType": _text(param.get("type")) or "any",
            **({"defaultValue": copy.deepcopy(default_value)} if default_present else {}),
            **({"required": True} if param.get("optional") is False else {}),
            **({"sealed": True} if param.get("disabled") is True else {}),
            "order": order,
            **({"help": param["description"]} if _text(param.get("description")) else {}),
        }
        result.append(migrated)
    return result


def _previews_from_v1(block: Mapping[str, Any]) -> list[dict[str, Any]]:
    previews = []
    for candidate in _array(block.get("nodes")):
        node = _object(candidate)
        node_id = _text(node.get("id"))
        if not node_id:
            continue
        for field_id, raw_param in _object(_object(node.get("data")).get("params")).items():
            display = _object(raw_param).get("display")
            media_type = _PREVIEW_MEDIA.get(display) if isinstance(display, str) else None
            if not media_type:
                continue
            previews.append(
                {
                    "nodeId": node_id,
                    "outputPortId": field_id,
                    "mediaType": media_type,
                    **({"primary": True} if not previews else {}),
                }
            )
    return previews


def _source_from_v1(block: Mapping[str, Any]) -> dict[str, Any]:
    origin = _object(block.get("origin"))
    if not origin:
        return {"kind": "user"}
    provider = _required_text(origin.get("provider"), "legacy User Node origin.provider")
    if provider not in {"diffusers", "transformers"}:
        raise CompositeMigrationError("legacy User Node origin.provider is unsupported.")
    source_kind = "hub_import" if origin.get("kind") == "hugging_face_hub_import" else "user"
    source: dict[str, Any] = {"kind": source_kind, "provider": provider, "library": provider}
    pairs = (
        ("libraryRevision", "libraryRevision"),
        ("pipelineClass", "pipelineClass"),
        ("workflowId", "workflow"),
        ("definitionId", "manifestDefinitionId"),
        ("contentHash", "manifestContentHash"),
        ("repo", "repository"),
        ("revision", "repositoryRevision"),
    )
    for source_key, target_key in pairs:
        if _text(origin.get(source_key)):
            source[target_key] = origin[source_key]
    if _text(origin.get("definitionId")) and _text(origin.get("contentHash")):
        source["parent"] = {
            "definitionId": origin["definitionId"],
            "contentHash": origin["contentHash"],
            "sourceKind": "diffusers_catalog" if provider == "diffusers" else "transformers_catalog",
        }
    return source


def migrate_user_block_definition_v1(value: Any) -> dict[str, Any]:
    """Convert one V1 reusable definition using the synchronized client adapter rules."""

    block = _object(value)
    if block.get("version") != 1 or isinstance(block.get("version"), bool):
        raise CompositeMigrationError("legacy User Node version must be 1.")
    definition_id = _required_text(block.get("id"), "legacy User Node.id")
    display_name = _required_text(block.get("name"), "legacy User Node.name")
    for field in ("nodes", "edges", "inputs", "outputs", "exposedParams"):
        if not isinstance(block.get(field), list):
            raise CompositeMigrationError(f"legacy User Node {field} must be a list.")
    controls_v1 = [_object(control) for control in block["exposedParams"]]
    graph = _graph_from_v1(block, controls_v1)
    source = _source_from_v1(block)
    boundary_mode = "explicit" if block.get("origin") else "derived"
    definition = {
        "schemaVersion": 2,
        "definitionId": definition_id,
        "displayName": display_name,
        "source": source,
        "graph": graph,
        "boundary": {
            "mode": boundary_mode,
            "inputs": [
                _port_from_legacy(port, f"legacy User Node inputs[{index}]")
                for index, port in enumerate(block["inputs"])
            ],
            "outputs": [
                _port_from_legacy(port, f"legacy User Node outputs[{index}]")
                for index, port in enumerate(block["outputs"])
            ],
            **(
                {
                    "derivation": {
                        "algorithmVersion": "legacy-user-block-boundary-v1",
                        "derivedAtDefinitionHash": graph["graphHash"],
                    }
                }
                if boundary_mode == "derived"
                else {}
            ),
        },
        "controls": _controls_from_v1(block, graph, controls_v1),
        "previews": _previews_from_v1(block),
        "ownership": {"kind": "user", "definitionMutable": True},
    }
    definition["contentHash"] = block_definition_content_hash_v2(definition)
    return validate_block_definition_v2(definition)


def _node_layout(node: Mapping[str, Any], label: str) -> dict[str, Any]:
    position = _object(node.get("position"))
    if not _finite(position.get("x")) or not _finite(position.get("y")):
        raise CompositeMigrationError(f"{label} position is missing or non-finite.")
    layout = {"x": position["x"], "y": position["y"]}
    for key in ("width", "height"):
        if node.get(key) is not None:
            if not _finite(node[key]) or node[key] <= 0:
                raise CompositeMigrationError(f"{label} {key} must be a positive finite number.")
            layout[key] = node[key]
    return layout


def _root_size(root: Mapping[str, Any]) -> dict[str, Any]:
    width = root.get("width")
    height = root.get("height")
    if not _finite(width) or width <= 0 or not _finite(height) or height <= 0:
        raise CompositeMigrationError("legacy User Node root must have a positive finite width and height.")
    return {"width": width, "height": height}


def _binding_port(ports: Sequence[dict[str, Any]], node_id: str, field_id: Any, label: str) -> dict[str, Any]:
    matches = [
        port
        for port in ports
        if _object(port.get("binding")).get("nodeId") == node_id
        and _object(port.get("binding")).get("fieldOrPortId") == field_id
    ]
    if len(matches) != 1:
        raise CompositeMigrationError(
            f"{label} cannot be preserved: expected exactly one public binding for {node_id}.{field_id}."
        )
    return matches[0]


def _child_semantics(nodes: Sequence[dict[str, Any]], root_id: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    children = [node for node in nodes if _object(node.get("data")).get("userBlockInstanceId") == root_id]
    child_to_semantic: dict[str, str] = {}
    semantic_ids: set[str] = set()
    for index, child in enumerate(children):
        child_id = _required_text(child.get("id"), f"legacy User Node {root_id} child[{index}].id")
        semantic_id = _required_text(
            _object(child.get("data")).get("userBlockSourceNodeId"),
            f"legacy User Node {root_id} child {child_id} source id",
        )
        if semantic_id in semantic_ids:
            raise CompositeMigrationError(f"legacy User Node {root_id} has duplicate semantic child {semantic_id}.")
        parent_id = child.get("parentId")
        if parent_id not in {None, root_id}:
            raise CompositeMigrationError(
                f"legacy User Node {root_id} child {child_id} uses an internal parent hierarchy that V2 forbids."
            )
        semantic_ids.add(semantic_id)
        child_to_semantic[child_id] = semantic_id
    return children, child_to_semantic


def _effective_graph_from_instance(
    definition: Mapping[str, Any],
    legacy_snapshot: Mapping[str, Any],
    root: Mapping[str, Any],
    nodes: Sequence[dict[str, Any]],
    edges: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], set[str], list[dict[str, Any]]]:
    root_id = _required_text(root.get("id"), "legacy User Node root.id")
    data = _object(root.get("data"))
    children, child_to_semantic = _child_semantics(nodes, root_id)
    expanded = bool(_object(data.get("uiState")).get("blockExpanded")) or bool(children)
    base_graph = _object(definition.get("graph"))
    base_nodes = {_object(node).get("nodeId"): _object(node) for node in _array(base_graph.get("nodes"))}

    if expanded:
        graph_nodes = []
        internal_layout = {}
        for child in children:
            child_id = _required_text(child.get("id"), "legacy User Node child.id")
            semantic_id = child_to_semantic[child_id]
            base = base_nodes.get(semantic_id, {})
            child_data = _object(child.get("data"))
            graph_nodes.append(
                {
                    "nodeId": semantic_id,
                    "nodeType": _text(child.get("type")) or _text(child_data.get("type")) or "custom",
                    "data": _strip_legacy_node_data(child),
                    **({"semanticRole": base["semanticRole"]} if _text(base.get("semanticRole")) else {}),
                    **(
                        {"upstreamBlockPath": base["upstreamBlockPath"]}
                        if _text(base.get("upstreamBlockPath"))
                        else {}
                    ),
                }
            )
            internal_layout[semantic_id] = _node_layout(child, f"legacy User Node child {child_id}")
        graph_edges = []
        for edge_index, edge in enumerate(edges):
            source_semantic = child_to_semantic.get(_text(edge.get("source")) or "")
            target_semantic = child_to_semantic.get(_text(edge.get("target")) or "")
            if not source_semantic or not target_semantic:
                continue
            edge_data = _object(edge.get("data"))
            if edge_data.get("userBlockBridge") is True:
                continue
            graph_edges.append(
                {
                    "edgeId": _required_text(edge.get("id"), f"workflow internal edge[{edge_index}].id"),
                    "sourceNodeId": source_semantic,
                    "sourcePortId": _required_text(
                        edge.get("sourceHandle"), f"workflow internal edge {edge.get('id')} sourceHandle"
                    ),
                    "targetNodeId": target_semantic,
                    "targetPortId": _required_text(
                        edge.get("targetHandle"), f"workflow internal edge {edge.get('id')} targetHandle"
                    ),
                }
            )
        if not graph_edges and graph_nodes and _array(base_graph.get("edges")):
            graph_edges = copy.deepcopy(base_graph["edges"])
        graph: dict[str, Any] = {"nodes": graph_nodes, "edges": graph_edges}
    else:
        graph = {
            "nodes": copy.deepcopy(_array(base_graph.get("nodes"))),
            "edges": copy.deepcopy(_array(base_graph.get("edges"))),
            **(
                {"executionOrder": copy.deepcopy(base_graph["executionOrder"])}
                if "executionOrder" in base_graph
                else {}
            ),
        }
        internal_layout = {}
        for index, legacy_node_value in enumerate(_array(legacy_snapshot.get("nodes"))):
            legacy_node = _object(legacy_node_value)
            node_id = _required_text(legacy_node.get("id"), f"legacy User Node snapshot node[{index}].id")
            layout = _node_layout(legacy_node, f"legacy User Node snapshot node {node_id}")
            layout["x"] += _LEGACY_CHILD_LEFT
            layout["y"] += _LEGACY_CHILD_TOP
            internal_layout[node_id] = layout

    graph["graphHash"] = block_graph_hash_v2(graph)
    owned_ids = {root_id, *child_to_semantic.keys()}
    return graph, internal_layout, owned_ids, children


def _instance_values(definition: Mapping[str, Any], root: Mapping[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    params = _object(_object(root.get("data")).get("params"))
    values: dict[str, Any] = {}
    value_bindings = []
    for port in _array(_object(definition.get("boundary")).get("inputs")):
        item = _object(port)
        value_bindings.append((item.get("portId"), _object(item.get("binding"))))
    for control in _array(definition.get("controls")):
        item = _object(control)
        binding = _object(item.get("binding"))
        value_bindings.append(
            (
                item.get("controlId"),
                {"nodeId": binding.get("nodeId"), "fieldOrPortId": binding.get("fieldId")},
            )
        )
    allowed = {value_id for value_id, _ in value_bindings}
    allowed.discard(None)
    for value_id in sorted(allowed):
        param = _object(params.get(value_id))
        if "value" in param:
            values[value_id] = copy.deepcopy(param["value"])

    graph_nodes = {_object(node).get("nodeId"): _object(node) for node in _array(graph.get("nodes"))}
    definition_nodes = {
        _object(node).get("nodeId"): _object(node) for node in _array(_object(definition.get("graph")).get("nodes"))
    }
    for value_id, binding in value_bindings:
        node_id = binding.get("nodeId")
        field_id = binding.get("fieldOrPortId")
        node = graph_nodes.get(node_id)
        if node is None or not isinstance(field_id, str) or not field_id:
            continue
        node_data = _object(node.get("data"))
        node_params = _object(node_data.get("params"))
        field = _object(node_params.get(field_id))
        if value_id not in values and "value" in field:
            values[value_id] = copy.deepcopy(field["value"])
        definition_field = _object(_object(_object(definition_nodes.get(node_id)).get("data")).get("params")).get(
            field_id
        )
        definition_field = _object(definition_field)
        if "value" in definition_field:
            field["value"] = copy.deepcopy(definition_field["value"])
        else:
            field.pop("value", None)
        node_params[field_id] = field
        node_data["params"] = node_params
        node["data"] = node_data

    # Declared input/control values belong exclusively in instance.values.
    # Only legacy output/preview state has no value slot and therefore needs
    # to remain on its effective source field as well as previewStates.
    bindings = []
    for port in _array(_object(definition.get("boundary")).get("outputs")):
        item = _object(port)
        bindings.append((item.get("portId"), _object(item.get("binding"))))
    for value_id, binding in bindings:
        root_param = _object(params.get(value_id))
        semantic_fields = {
            key: copy.deepcopy(value) for key, value in root_param.items() if key not in _PARAM_PRESENTATION_FIELDS
        }
        if not semantic_fields:
            continue
        node = graph_nodes.get(binding.get("nodeId"))
        if node is None:
            continue
        node_data = _object(node.get("data"))
        node_params = _object(node_data.get("params"))
        field_id = binding.get("fieldOrPortId")
        if not isinstance(field_id, str) or not field_id:
            continue
        node_params[field_id] = {**_object(node_params.get(field_id)), **semantic_fields}
        node_data["params"] = node_params
        node["data"] = node_data
    graph["graphHash"] = block_graph_hash_v2(graph)
    return values


def _preview_states(definition: Mapping[str, Any], root: Mapping[str, Any]) -> list[dict[str, Any]]:
    params = _object(_object(root.get("data")).get("params"))
    result = []
    for binding in _array(definition.get("previews")):
        preview = _object(binding)
        match = None
        for param in params.values():
            candidate = _object(param)
            options = _object(candidate.get("fieldOptions"))
            if options.get("userBlockSourceNodeId") == preview.get("nodeId") and options.get(
                "userBlockSourceFieldKey"
            ) == preview.get("outputPortId"):
                match = candidate
                break
        state = {"binding": copy.deepcopy(preview), "status": "idle"}
        if match and _text(match.get("value")):
            state.update({"mediaReference": match["value"], "status": "complete"})
        result.append(state)
    return result


def _rewrite_workflow_edges(
    edges: Sequence[dict[str, Any]],
    root_id: str,
    owned_ids: set[str],
    child_to_semantic: Mapping[str, str],
    definition: Mapping[str, Any],
) -> list[dict[str, Any]]:
    boundary = _object(definition.get("boundary"))
    inputs = [_object(port) for port in _array(boundary.get("inputs"))]
    outputs = [_object(port) for port in _array(boundary.get("outputs"))]
    result = []
    for edge in edges:
        source = _text(edge.get("source"))
        target = _text(edge.get("target"))
        source_owned = source in owned_ids
        target_owned = target in owned_ids
        if source_owned and target_owned:
            edge_data = _object(edge.get("data"))
            if source in child_to_semantic and target in child_to_semantic:
                continue
            if edge_data.get("userBlockBridge") is True:
                continue
            raise CompositeMigrationError(
                f"workflow edge {edge.get('id')} crosses a legacy root and child without a bridge receipt."
            )
        migrated = copy.deepcopy(edge)
        if source in child_to_semantic:
            port = _binding_port(
                outputs,
                child_to_semantic[source],
                edge.get("sourceHandle"),
                f"workflow edge {edge.get('id')} source",
            )
            migrated["source"] = root_id
            migrated["sourceHandle"] = port["portId"]
        elif source == root_id:
            if len([port for port in outputs if port.get("portId") == edge.get("sourceHandle")]) != 1:
                raise CompositeMigrationError(
                    f"workflow edge {edge.get('id')} uses an unknown legacy root output handle."
                )
        if target in child_to_semantic:
            port = _binding_port(
                inputs,
                child_to_semantic[target],
                edge.get("targetHandle"),
                f"workflow edge {edge.get('id')} target",
            )
            migrated["target"] = root_id
            migrated["targetHandle"] = port["portId"]
        elif target == root_id:
            if len([port for port in inputs if port.get("portId") == edge.get("targetHandle")]) != 1:
                raise CompositeMigrationError(
                    f"workflow edge {edge.get('id')} uses an unknown legacy root input handle."
                )
        result.append(migrated)
    return result


def _convert_v1_root(
    root: Mapping[str, Any], nodes: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    root_id = _required_text(root.get("id"), "legacy User Node root.id")
    root_data = _object(root.get("data"))
    snapshot = _object(root_data.get("userBlockSnapshot"))
    if not snapshot:
        raise CompositeMigrationError(f"legacy User Node {root_id} has no embedded snapshot.")
    definition = migrate_user_block_definition_v1(snapshot)
    effective_graph, internal_layout, owned_ids, children = _effective_graph_from_instance(
        definition, snapshot, root, nodes, edges
    )
    values = _instance_values(definition, root, effective_graph)
    definition_graph_hash = _object(definition.get("graph")).get("graphHash")
    effective_hash = effective_graph["graphHash"]
    if effective_hash != definition_graph_hash:
        customization_state = "structure_changed"
    else:
        default_values = {
            control.get("controlId"): control.get("defaultValue")
            for control in _array(definition.get("controls"))
            if "defaultValue" in _object(control)
        }
        customization_state = (
            "parameters_changed"
            if any(
                value_id not in default_values or default_values[value_id] != value
                for value_id, value in values.items()
            )
            else "unchanged"
        )
    instance = {
        "schemaVersion": 2,
        "instanceId": root_id,
        "definitionRef": {
            "definitionId": definition["definitionId"],
            "contentHash": definition["contentHash"],
        },
        "definitionSnapshot": definition,
        "effectiveGraph": effective_graph,
        "effectiveInterface": {
            "boundary": copy.deepcopy(definition["boundary"]),
            "controls": copy.deepcopy(definition["controls"]),
            "baseInterfaceHash": block_interface_hash_v2(definition),
            "effectiveInterfaceHash": block_interface_hash_v2(definition),
        },
        "values": values,
        "customization": {
            "state": customization_state,
            "baseGraphHash": definition_graph_hash,
            "effectiveGraphHash": effective_hash,
        },
        "presentation": {
            "expanded": bool(_object(root_data.get("uiState")).get("blockExpanded")) or bool(children),
            "position": copy.deepcopy(_node_layout(root, f"legacy User Node root {root_id}")),
            "size": _root_size(root),
            "internalLayout": internal_layout,
        },
        "previewStates": _preview_states(definition, root),
        "authorities": [],
    }
    instance["presentation"]["position"] = {
        "x": instance["presentation"]["position"]["x"],
        "y": instance["presentation"]["position"]["y"],
    }
    converted_root = {
        "id": root_id,
        "type": "block",
        "position": copy.deepcopy(instance["presentation"]["position"]),
        "width": instance["presentation"]["size"]["width"],
        "height": instance["presentation"]["size"]["height"],
        "data": {
            "type": "block",
            "module": "MoDiff",
            "action": "BlockV2",
            "label": definition["displayName"],
            "category": "Hub imports" if definition["source"]["kind"] == "hub_import" else "User Nodes",
            "params": {},
            "resizable": True,
            "blockInstanceV2": instance,
        },
    }
    child_to_semantic = {
        _required_text(child.get("id"), "legacy User Node child.id"): _required_text(
            _object(child.get("data")).get("userBlockSourceNodeId"), "legacy User Node child source id"
        )
        for child in children
    }
    migrated_edges = _rewrite_workflow_edges(edges, root_id, owned_ids, child_to_semantic, definition)
    return converted_root, migrated_edges, owned_ids - {root_id}


def _is_legacy_cluster_root(node: Mapping[str, Any]) -> bool:
    data = _object(node.get("data"))
    return (
        node.get("type") == "cluster"
        or data.get("type") == "cluster"
        or data.get("huggingFaceClusterRole") == "root"
        or data.get("huggingFaceClusterInstance") is not None
    )


def _legacy_cluster_children(
    root_id: str, nodes: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    children = []
    for node in nodes:
        if node.get("id") == root_id:
            continue
        data = _object(node.get("data"))
        if node.get("parentId") == root_id or data.get("huggingFaceClusterInstanceId") == root_id:
            children.append(node)
    return sorted(children, key=lambda node: str(node.get("id") or ""))


def legacy_registered_cluster_composite_hash(
    document: Mapping[str, Any], instance_id: str
) -> str:
    """Hash one exact legacy Cluster root, projections, and touching edges.

    This is a compiler handoff receipt, not an execution hash.  It binds a
    supplied V2 instance to the precise legacy material the compiler reviewed.
    """

    snapshot = _object(document.get("snapshot"))
    nodes = [_object(node) for node in _array(snapshot.get("nodes"))]
    edges = [_object(edge) for edge in _array(snapshot.get("edges"))]
    root = next((node for node in nodes if node.get("id") == instance_id), None)
    if root is None or not _is_legacy_cluster_root(root):
        raise CompositeMigrationError(f"legacy registered Cluster {instance_id} was not found.")
    children = _legacy_cluster_children(instance_id, nodes)
    owned_ids = {instance_id, *(str(child.get("id")) for child in children)}
    touching_edges = [
        edge for edge in edges if edge.get("source") in owned_ids or edge.get("target") in owned_ids
    ]
    material = {
        "root": copy.deepcopy(root),
        "ownedNodes": copy.deepcopy(children),
        "touchingEdges": sorted(
            copy.deepcopy(touching_edges), key=lambda edge: str(edge.get("id") or "")
        ),
    }
    return _sha256_json(material)


def _ordered_mapping_items(
    value: Any,
    label: str,
    *,
    required: set[str],
    key_fields: Sequence[str],
    optional: set[str] | None = None,
) -> list[dict[str, Any]]:
    values = _array(value)
    if value is not None and not isinstance(value, list):
        raise CompositeMigrationError(f"{label} must be a list.")
    result = []
    previous_key: tuple[str, ...] | None = None
    for index, candidate in enumerate(values):
        item = _object(candidate)
        _exact_keys(item, f"{label}[{index}]", required=required, optional=optional)
        key = tuple(_required_text(item.get(field), f"{label}[{index}].{field}") for field in key_fields)
        if previous_key is not None and key <= previous_key:
            raise CompositeMigrationError(f"{label} must be unique and canonically ordered.")
        previous_key = key
        result.append(copy.deepcopy(item))
    return result


def _normalize_compiler_supplement(value: Any) -> dict[str, Any]:
    if value is None:
        return {
            "schemaVersion": COMPILER_SUPPLEMENT_SCHEMA_VERSION,
            "kind": _COMPILER_SUPPLEMENT_KIND,
            "compilerOutputs": [],
        }
    supplement = _object(value)
    _exact_keys(
        supplement,
        "compiler supplement",
        required={"schemaVersion", "kind", "compilerOutputs"},
    )
    if supplement.get("schemaVersion") != COMPILER_SUPPLEMENT_SCHEMA_VERSION:
        raise CompositeMigrationError("compiler supplement schemaVersion must be 1.")
    if supplement.get("kind") != _COMPILER_SUPPLEMENT_KIND:
        raise CompositeMigrationError(f"compiler supplement kind must be {_COMPILER_SUPPLEMENT_KIND}.")
    outputs_value = supplement.get("compilerOutputs")
    if not isinstance(outputs_value, list):
        raise CompositeMigrationError("compiler supplement compilerOutputs must be a list.")
    if len(_canonical_json(supplement).encode("utf-8")) > 32 * 1024 * 1024:
        raise CompositeMigrationError("compiler supplement exceeds the 32 MiB request limit.")
    outputs = []
    previous_path: str | None = None
    for index, output_value in enumerate(outputs_value):
        output = _object(output_value)
        label = f"compiler supplement compilerOutputs[{index}]"
        _exact_keys(
            output,
            label,
            required={"sourcePath", "sourceSha256", "conversions"},
        )
        source_path = _required_text(output.get("sourcePath"), f"{label}.sourcePath")
        _required_text(output.get("sourceSha256"), f"{label}.sourceSha256")
        path = PurePosixPath(source_path)
        if len(path.parts) != 2 or path.parts[0] != "user-workflows" or path.suffix != ".json":
            raise CompositeMigrationError(f"{label}.sourcePath must name one user-workflows JSON file.")
        if previous_path is not None and source_path <= previous_path:
            raise CompositeMigrationError("compiler supplement outputs must be unique and ordered by sourcePath.")
        previous_path = source_path
        conversions_value = output.get("conversions")
        if not isinstance(conversions_value, list) or not conversions_value:
            raise CompositeMigrationError(f"{label}.conversions must be a non-empty list.")
        conversions = []
        previous_id: str | None = None
        for conversion_index, conversion_value in enumerate(conversions_value):
            conversion = _object(conversion_value)
            conversion_label = f"{label}.conversions[{conversion_index}]"
            _exact_keys(
                conversion,
                conversion_label,
                required={
                    "legacyInstanceId",
                    "legacyCompositeHash",
                    "admissionId",
                    "compiledDefinitionContentHash",
                    "compiledDefinitionCanonicalSha256",
                    "blockInstanceV2",
                    "ownedNodeMappings",
                    "portMappings",
                    "valueMappings",
                    "previewMappings",
                    "absorbedInternalEdgeIds",
                },
                optional={"semanticEquivalenceReceipt", "historicalCompilerMapping"},
            )
            instance_id = _required_text(
                conversion.get("legacyInstanceId"), f"{conversion_label}.legacyInstanceId"
            )
            for field in (
                "legacyCompositeHash",
                "admissionId",
                "compiledDefinitionContentHash",
                "compiledDefinitionCanonicalSha256",
            ):
                _required_text(conversion.get(field), f"{conversion_label}.{field}")
            if previous_id is not None and instance_id <= previous_id:
                raise CompositeMigrationError(f"{label}.conversions must be unique and ordered by legacyInstanceId.")
            previous_id = instance_id
            if not isinstance(conversion.get("blockInstanceV2"), dict):
                raise CompositeMigrationError(f"{conversion_label}.blockInstanceV2 must be an object.")
            for field in (
                "ownedNodeMappings",
                "portMappings",
                "valueMappings",
                "previewMappings",
                "absorbedInternalEdgeIds",
            ):
                if not isinstance(conversion.get(field), list):
                    raise CompositeMigrationError(f"{conversion_label}.{field} must be a list.")
            if conversion.get("semanticEquivalenceReceipt") is not None:
                reference = _object(conversion.get("semanticEquivalenceReceipt"))
                _exact_keys(
                    reference,
                    f"{conversion_label}.semanticEquivalenceReceipt",
                    required={"receiptId", "receiptHash"},
                )
                _required_text(
                    reference.get("receiptId"),
                    f"{conversion_label}.semanticEquivalenceReceipt.receiptId",
                )
                _required_text(
                    reference.get("receiptHash"),
                    f"{conversion_label}.semanticEquivalenceReceipt.receiptHash",
                )
            if conversion.get("historicalCompilerMapping") is not None:
                reference = _object(conversion.get("historicalCompilerMapping"))
                _exact_keys(
                    reference,
                    f"{conversion_label}.historicalCompilerMapping",
                    required={"mappingId", "mappingHash"},
                )
                _required_text(
                    reference.get("mappingId"),
                    f"{conversion_label}.historicalCompilerMapping.mappingId",
                )
                _required_text(
                    reference.get("mappingHash"),
                    f"{conversion_label}.historicalCompilerMapping.mappingHash",
                )
            if (
                conversion.get("semanticEquivalenceReceipt") is not None
                and conversion.get("historicalCompilerMapping") is not None
            ):
                raise CompositeMigrationError(
                    f"{conversion_label} must not combine semantic-equivalence and archived compiler-mapping authority."
                )
            conversions.append(copy.deepcopy(conversion))
        outputs.append({**copy.deepcopy(output), "conversions": conversions})
    return {
        "schemaVersion": COMPILER_SUPPLEMENT_SCHEMA_VERSION,
        "kind": _COMPILER_SUPPLEMENT_KIND,
        "compilerOutputs": outputs,
    }


def _legacy_value_sources(
    root: Mapping[str, Any], children: Sequence[dict[str, Any]]
) -> dict[tuple[str, str, str], Any]:
    result: dict[tuple[str, str, str], Any] = {}
    for node in sorted([root, *children], key=lambda item: str(item.get("id") or "")):
        node_id = _required_text(node.get("id"), "legacy Cluster node id")
        for field_id, param_value in sorted(_object(_object(node.get("data")).get("params")).items()):
            param = _object(param_value)
            if "value" in param and param.get("display") != "output":
                result[("node_param", node_id, field_id)] = copy.deepcopy(param["value"])
    legacy_instance = _object(_object(root.get("data")).get("huggingFaceClusterInstance"))
    for source_kind, values in (
        ("instance_parameter_override", _object(legacy_instance.get("parameterOverrides"))),
        (
            "execution_parameter_override",
            _object(_object(legacy_instance.get("execution")).get("parameterOverrides")),
        ),
    ):
        for field_id, field_value in sorted(values.items()):
            result[(source_kind, str(root.get("id")), field_id)] = copy.deepcopy(field_value)
    return result


def _legacy_preview_sources(
    root: Mapping[str, Any], children: Sequence[dict[str, Any]]
) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for node in sorted([root, *children], key=lambda item: str(item.get("id") or "")):
        node_id = _required_text(node.get("id"), "legacy Cluster node id")
        for field_id, param_value in sorted(_object(_object(node.get("data")).get("params")).items()):
            param = _object(param_value)
            if param.get("display") != "output" or param.get("value") is None:
                continue
            media_reference = param.get("value")
            if not isinstance(media_reference, str) or not media_reference:
                raise CompositeMigrationError(
                    f"legacy output {node_id}.{field_id} is not a persistable media reference."
                )
            result[(node_id, field_id)] = media_reference
    return result


def _mapped_v2_value(instance: Mapping[str, Any], mapping: Mapping[str, Any], label: str) -> Any:
    target_kind = _required_text(mapping.get("targetKind"), f"{label}.targetKind")
    if target_kind == "instance_value":
        _exact_keys(
            mapping,
            label,
            required={"sourceKind", "sourceNodeId", "sourceFieldId", "targetKind", "targetValueId"},
        )
        target_id = _required_text(mapping.get("targetValueId"), f"{label}.targetValueId")
        values = _object(instance.get("values"))
        if target_id not in values:
            raise CompositeMigrationError(f"{label} targets missing BlockInstanceV2 value {target_id}.")
        return values[target_id]
    if target_kind == "graph_param":
        _exact_keys(
            mapping,
            label,
            required={
                "sourceKind",
                "sourceNodeId",
                "sourceFieldId",
                "targetKind",
                "targetNodeId",
                "targetFieldId",
            },
        )
        node_id = _required_text(mapping.get("targetNodeId"), f"{label}.targetNodeId")
        field_id = _required_text(mapping.get("targetFieldId"), f"{label}.targetFieldId")
        node = next(
            (item for item in _array(_object(instance.get("effectiveGraph")).get("nodes")) if _object(item).get("nodeId") == node_id),
            None,
        )
        param = _object(_object(_object(node).get("data")).get("params")).get(field_id)
        if not isinstance(param, dict) or "value" not in param:
            raise CompositeMigrationError(f"{label} targets missing graph parameter {node_id}.{field_id}.")
        return param["value"]
    raise CompositeMigrationError(f"{label}.targetKind is unsupported.")


def _category_for_registered_source(source: Mapping[str, Any]) -> str:
    return "Diffusers" if source.get("kind") == "diffusers_catalog" else "Transformers"


def _validate_registered_compiler_instance(
    root: Mapping[str, Any], conversion: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    try:
        instance = validate_block_instance_v2(conversion.get("blockInstanceV2"))
    except (TypeError, ValueError) as exc:
        raise CompositeMigrationError(f"compiled BlockInstanceV2 is invalid: {exc}") from exc
    root_id = _required_text(root.get("id"), "legacy Cluster root.id")
    if instance["instanceId"] != root_id or conversion.get("legacyInstanceId") != root_id:
        raise CompositeMigrationError("compiled BlockInstanceV2 must preserve the legacy Cluster instance id.")
    definition = instance["definitionSnapshot"]
    source = _object(definition.get("source"))
    if source.get("kind") not in _CATALOG_SOURCE_KINDS:
        raise CompositeMigrationError("compiled definition must be a registered Diffusers or Transformers catalog definition.")
    admission_id = _required_text(source.get("executionAdmissionId"), "compiled definition executionAdmissionId")
    canonical_sha = block_definition_canonical_sha256_v2(definition)
    declared = (
        conversion.get("compiledDefinitionContentHash"),
        conversion.get("compiledDefinitionCanonicalSha256"),
    )
    actual = (definition.get("contentHash"), canonical_sha)
    if conversion.get("admissionId") != admission_id or declared != actual:
        raise CompositeMigrationError("compiler output identity does not match its embedded registered definition.")
    if _registered_definition_pins().get(admission_id) != actual:
        raise CompositeMigrationError("compiled registered definition is not pinned by the backend admission registry.")
    legacy_instance = _object(_object(root.get("data")).get("huggingFaceClusterInstance"))
    legacy_definition = _object(legacy_instance.get("definition"))
    legacy_execution = _object(legacy_instance.get("execution"))
    legacy_identity = {
        "manifestDefinitionId": _text(legacy_definition.get("id")),
        "libraryRevision": _text(legacy_definition.get("libraryRevision")),
        "manifestContentHash": _text(legacy_definition.get("contentHash")),
        "executionAdmissionId": _text(legacy_execution.get("admissionId")),
        "studioExecutionSpec": copy.deepcopy(_object(legacy_execution.get("studioExecutionSpec"))),
    }
    current_identity = {
        "manifestDefinitionId": source.get("manifestDefinitionId"),
        "libraryRevision": source.get("libraryRevision"),
        "manifestContentHash": source.get("manifestContentHash"),
        "executionAdmissionId": admission_id,
    }
    exact_current = all(legacy_identity.get(key) == value for key, value in current_identity.items())
    receipt_reference = conversion.get("semanticEquivalenceReceipt")
    mapping_reference = conversion.get("historicalCompilerMapping")
    semantic_equivalence: dict[str, Any] | None = None
    historical_compiler_mapping: dict[str, Any] | None = None
    if exact_current:
        if receipt_reference is not None or mapping_reference is not None:
            raise CompositeMigrationError(
                "an exact-current Cluster conversion must not claim historical review authority."
            )
    else:
        if (receipt_reference is None) == (mapping_reference is None):
            raise CompositeMigrationError(
                "historical Cluster conversion requires exactly one checked-in semantic-equivalence receipt or archived compiler mapping."
            )
        if receipt_reference is not None:
            reference = _object(receipt_reference)
            receipt = _semantic_equivalence_receipt_by_reference(
                _required_text(reference.get("receiptId"), "semantic-equivalence receipt id"),
                _required_text(reference.get("receiptHash"), "semantic-equivalence receipt hash"),
            )
            if receipt is None:
                raise CompositeMigrationError(
                    "semantic-equivalence receipt is not present in the checked-in review ledger."
                )
            historical = _object(receipt.get("historical"))
            if any(historical.get(key) != value for key, value in legacy_identity.items()):
                raise CompositeMigrationError(
                    "semantic-equivalence receipt does not bind the exact historical manifest and execution receipt."
                )
            destination = _object(receipt.get("destination"))
            semantic_equivalence = _semantic_equivalence_authority(receipt)
        else:
            reference = _object(mapping_reference)
            mapping = _historical_compiler_mapping_by_reference(
                _required_text(reference.get("mappingId"), "historical compiler mapping id"),
                _required_text(reference.get("mappingHash"), "historical compiler mapping hash"),
            )
            if mapping is None:
                raise CompositeMigrationError(
                    "historical compiler mapping is not present in the checked-in review ledger."
                )
            historical = _object(mapping.get("historical"))
            if any(historical.get(key) != value for key, value in legacy_identity.items()):
                raise CompositeMigrationError(
                    "historical compiler mapping does not bind the exact archived manifest and execution receipt."
                )
            destination = _object(mapping.get("destination"))
            historical_compiler_mapping = _historical_compiler_mapping_authority(mapping)
        expected_destination = {
            "manifestDefinitionId": source.get("manifestDefinitionId"),
            "libraryRevision": source.get("libraryRevision"),
            "manifestContentHash": source.get("manifestContentHash"),
            "executionAdmissionId": admission_id,
            "blockDefinitionId": definition.get("definitionId"),
            "blockDefinitionContentHash": definition.get("contentHash"),
            "blockDefinitionCanonicalSha256": canonical_sha,
            "executionGraphHash": _object(definition.get("graph")).get("graphHash"),
            "interfaceHash": block_interface_hash_v2(definition),
        }
        if destination != expected_destination:
            raise CompositeMigrationError(
                "historical review authority is stale for the compiled BlockDefinitionV2 graph or interface."
            )
    if legacy_instance.get("structuralFork") is not None:
        raise CompositeMigrationError("a structurally forked legacy Cluster requires User Node migration, not registered conversion.")
    if instance["customization"]["state"] == "structure_changed":
        raise CompositeMigrationError("registered compiler migration does not accept a structurally customized V2 instance.")
    if instance["effectiveGraph"]["graphHash"] != definition["graph"]["graphHash"]:
        raise CompositeMigrationError("registered compiler migration requires the exact pinned definition graph.")
    if instance["authorities"]:
        raise CompositeMigrationError("compiler migration must not carry execution authorities; the runtime reissues them.")
    root_layout = _node_layout(root, f"legacy Cluster root {root_id}")
    expected_position = {"x": root_layout["x"], "y": root_layout["y"]}
    expected_size = _root_size(root)
    if instance["presentation"]["position"] != expected_position or instance["presentation"]["size"] != expected_size:
        raise CompositeMigrationError("compiled BlockInstanceV2 must preserve the root position and size exactly.")
    expected_expanded = bool(_object(legacy_instance.get("presentation")).get("expanded"))
    if instance["presentation"]["expanded"] != expected_expanded:
        raise CompositeMigrationError("compiled BlockInstanceV2 must preserve the expanded presentation state.")
    return instance, semantic_equivalence, historical_compiler_mapping


def _port_binding_targets(port: Mapping[str, Any]) -> set[tuple[str, str]]:
    binding = _object(port.get("binding"))
    result = {(str(binding.get("nodeId")), str(binding.get("fieldOrPortId")))}
    result.update(
        (str(_object(mirror).get("nodeId")), str(_object(mirror).get("fieldOrPortId")))
        for mirror in _array(port.get("mirrorBindings"))
    )
    return result


def _legacy_execution_node_id(instance_id: str, semantic_role: str) -> str:
    """Reproduce the persisted legacy execution-child identity exactly.

    Collapsed legacy Clusters may retain root port/preview metadata while their
    presentation children are absent. This deterministic identity is the only
    admitted bridge from that metadata to a pinned V2 semantic role; labels,
    array positions, and model-specific inference are never used.
    """

    encoded_role = quote(semantic_role, safe="-_.!~*'()")
    return f"{instance_id}__diffusers.cluster-execution:{encoded_role}"


def _convert_registered_cluster_root(
    document: Mapping[str, Any],
    conversion: Mapping[str, Any],
    *,
    legacy_source_document: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    converted = copy.deepcopy(_object(document))
    snapshot = _object(converted.get("snapshot"))
    if not isinstance(snapshot.get("nodes"), list) or not isinstance(snapshot.get("edges"), list):
        raise CompositeMigrationError("saved workflow requires snapshot.nodes and snapshot.edges lists.")
    nodes = [copy.deepcopy(_object(node)) for node in snapshot["nodes"]]
    edges = [copy.deepcopy(_object(edge)) for edge in snapshot["edges"]]
    root_id = _required_text(conversion.get("legacyInstanceId"), "compiler conversion legacyInstanceId")
    root = next((node for node in nodes if node.get("id") == root_id), None)
    if root is None or not _is_legacy_cluster_root(root):
        raise CompositeMigrationError(f"legacy registered Cluster {root_id} was not found.")
    composite_hash = legacy_registered_cluster_composite_hash(
        legacy_source_document or converted,
        root_id,
    )
    if conversion.get("legacyCompositeHash") != composite_hash:
        raise CompositeMigrationError("compiler output was produced from a different legacy Cluster snapshot.")
    instance, semantic_equivalence, historical_compiler_mapping = (
        _validate_registered_compiler_instance(root, conversion)
    )
    children = _legacy_cluster_children(root_id, nodes)
    child_ids = {_required_text(child.get("id"), "legacy Cluster child.id") for child in children}
    owned_ids = {root_id, *child_ids}

    node_mappings = _ordered_mapping_items(
        conversion.get("ownedNodeMappings"),
        "compiler conversion ownedNodeMappings",
        required={"legacyNodeId", "semanticNodeId"},
        key_fields=("legacyNodeId", "semanticNodeId"),
    )
    mapped_child_ids = {mapping["legacyNodeId"] for mapping in node_mappings}
    semantic_ids = [mapping["semanticNodeId"] for mapping in node_mappings]
    if mapped_child_ids != child_ids or len(semantic_ids) != len(set(semantic_ids)):
        raise CompositeMigrationError(
            "compiler output must map every legacy projection node exactly once without merging nodes."
        )
    graph_node_ids = {
        _required_text(_object(node).get("nodeId"), "compiled graph nodeId")
        for node in _array(_object(instance.get("effectiveGraph")).get("nodes"))
    }
    if not set(semantic_ids).issubset(graph_node_ids):
        raise CompositeMigrationError("compiler output maps a legacy projection to an unknown semantic node.")
    semantic_by_legacy = {mapping["legacyNodeId"]: mapping["semanticNodeId"] for mapping in node_mappings}
    semantic_targets_by_legacy = dict(semantic_by_legacy)
    semantic_roles: set[str] = set()
    for graph_node_value in _array(_object(instance.get("effectiveGraph")).get("nodes")):
        graph_node = _object(graph_node_value)
        semantic_role = _text(graph_node.get("semanticRole"))
        if not semantic_role:
            continue
        if semantic_role in semantic_roles:
            raise CompositeMigrationError(
                f"compiled graph has duplicate semantic role {semantic_role}; legacy endpoint ownership is ambiguous."
            )
        semantic_roles.add(semantic_role)
        legacy_node_id = _legacy_execution_node_id(root_id, semantic_role)
        semantic_node_id = _required_text(graph_node.get("nodeId"), "compiled graph nodeId")
        previous = semantic_targets_by_legacy.get(legacy_node_id)
        if previous is not None and previous != semantic_node_id:
            raise CompositeMigrationError(
                f"legacy execution endpoint {legacy_node_id} maps to conflicting semantic nodes."
            )
        semantic_targets_by_legacy[legacy_node_id] = semantic_node_id
    child_by_id = {str(child.get("id")): child for child in children}
    internal_layout = _object(_object(instance.get("presentation")).get("internalLayout"))
    for legacy_id, semantic_id in semantic_by_legacy.items():
        child = child_by_id[legacy_id]
        expected = _node_layout(child, f"legacy Cluster child {legacy_id}")
        width = child.get("width")
        height = child.get("height")
        if _finite(width) and width > 0:
            expected["width"] = width
        if _finite(height) and height > 0:
            expected["height"] = height
        if _object(internal_layout.get(semantic_id)) != expected:
            raise CompositeMigrationError(
                f"compiled BlockInstanceV2 must preserve layout for projection node {legacy_id}."
            )

    port_mappings = _ordered_mapping_items(
        conversion.get("portMappings"),
        "compiler conversion portMappings",
        required={"direction", "legacyNodeId", "legacyPortId", "v2PortId"},
        key_fields=("direction", "legacyNodeId", "legacyPortId", "v2PortId"),
    )
    ports_by_direction = {
        "input": {port["portId"]: port for port in instance["effectiveInterface"]["boundary"]["inputs"]},
        "output": {port["portId"]: port for port in instance["effectiveInterface"]["boundary"]["outputs"]},
    }
    mapped_ports: dict[tuple[str, str, str], str] = {}
    for index, mapping in enumerate(port_mappings):
        direction = mapping["direction"]
        if direction not in {"input", "output"}:
            raise CompositeMigrationError(f"compiler conversion portMappings[{index}].direction is unsupported.")
        legacy_node_id = mapping["legacyNodeId"]
        legacy_port_id = mapping["legacyPortId"]
        if legacy_node_id not in owned_ids:
            raise CompositeMigrationError("compiler port mapping references a node outside the legacy Cluster.")
        port = ports_by_direction[direction].get(mapping["v2PortId"])
        if port is None:
            raise CompositeMigrationError("compiler port mapping references an unknown V2 public port.")
        if legacy_node_id == root_id:
            legacy_param = _object(_object(_object(root.get("data")).get("params")).get(legacy_port_id))
            options = _object(legacy_param.get("fieldOptions"))
            if options.get("huggingFaceClusterPortDirection") != direction:
                raise CompositeMigrationError("compiler root port mapping direction does not match legacy metadata.")
            physical_node_id = _required_text(
                options.get("huggingFaceClusterPortNodeId"), "legacy Cluster port node id"
            )
            semantic_node_id = semantic_targets_by_legacy.get(physical_node_id)
            field_id = _required_text(
                options.get("huggingFaceClusterPortField"), "legacy Cluster port field"
            )
        else:
            semantic_node_id = semantic_by_legacy.get(legacy_node_id)
            field_id = legacy_port_id
        if semantic_node_id is None or (semantic_node_id, field_id) not in _port_binding_targets(port):
            raise CompositeMigrationError("compiler public port does not bind the exact mapped legacy endpoint.")
        key = (direction, legacy_node_id, legacy_port_id)
        if key in mapped_ports:
            raise CompositeMigrationError("compiler port mappings contain an ambiguous legacy endpoint.")
        mapped_ports[key] = mapping["v2PortId"]

    declared_root_ports = set()
    for field_id, param_value in sorted(_object(_object(root.get("data")).get("params")).items()):
        options = _object(_object(param_value).get("fieldOptions"))
        direction = options.get("huggingFaceClusterPortDirection")
        if direction in {"input", "output"}:
            declared_root_ports.add((direction, root_id, field_id))
    if not declared_root_ports.issubset(mapped_ports):
        raise CompositeMigrationError("compiler output must map every declared legacy Cluster public port.")

    source_values = _legacy_value_sources(root, children)
    value_mappings = _ordered_mapping_items(
        conversion.get("valueMappings"),
        "compiler conversion valueMappings",
        required={"sourceKind", "sourceNodeId", "sourceFieldId", "targetKind"},
        key_fields=("sourceKind", "sourceNodeId", "sourceFieldId", "targetKind"),
        optional={"targetValueId", "targetNodeId", "targetFieldId"},
    )
    mapped_source_keys: set[tuple[str, str, str]] = set()
    for index, mapping in enumerate(value_mappings):
        if (
            historical_compiler_mapping is not None
            and mapping["targetKind"] != "instance_value"
        ):
            raise CompositeMigrationError(
                "historical compiler mapping may preserve values only through BlockInstanceV2 values; "
                "it must not rewrite the immutable destination definition graph."
            )
        source_key = (
            mapping["sourceKind"],
            mapping["sourceNodeId"],
            mapping["sourceFieldId"],
        )
        if source_key not in source_values or source_key in mapped_source_keys:
            raise CompositeMigrationError("compiler value mapping is unknown or duplicates a legacy value source.")
        mapped_value = _mapped_v2_value(instance, mapping, f"compiler conversion valueMappings[{index}]")
        if _canonical_json(mapped_value) != _canonical_json(source_values[source_key]):
            raise CompositeMigrationError("compiler value mapping changes a persisted prompt or parameter value.")
        mapped_source_keys.add(source_key)
    if mapped_source_keys != set(source_values):
        missing = sorted(set(source_values) - mapped_source_keys)
        raise CompositeMigrationError(
            "compiler output does not preserve every legacy prompt/parameter value: "
            + ", ".join("/".join(key) for key in missing)
        )

    preview_sources = _legacy_preview_sources(root, children)
    preview_mappings = _ordered_mapping_items(
        conversion.get("previewMappings"),
        "compiler conversion previewMappings",
        required={"sourceNodeId", "sourceFieldId", "targetNodeId", "targetOutputPortId"},
        key_fields=("sourceNodeId", "sourceFieldId", "targetNodeId", "targetOutputPortId"),
    )
    mapped_preview_keys: set[tuple[str, str]] = set()
    preview_states = _array(instance.get("previewStates"))
    for mapping in preview_mappings:
        source_key = (mapping["sourceNodeId"], mapping["sourceFieldId"])
        if source_key not in preview_sources or source_key in mapped_preview_keys:
            raise CompositeMigrationError(
                "compiler preview mapping is unknown or duplicates a legacy output reference."
            )
        if mapping["sourceNodeId"] == root_id:
            legacy_param = _object(
                _object(_object(root.get("data")).get("params")).get(mapping["sourceFieldId"])
            )
            options = _object(legacy_param.get("fieldOptions"))
            physical_node_id = _required_text(
                options.get("huggingFaceClusterPreviewSourceNodeId")
                or options.get("huggingFaceClusterPortNodeId"),
                "legacy Cluster preview source node id",
            )
            expected_node_id = semantic_targets_by_legacy.get(physical_node_id)
            expected_output_id = _required_text(
                options.get("huggingFaceClusterPreviewSourceField")
                or options.get("huggingFaceClusterPortField"),
                "legacy Cluster preview source field",
            )
        else:
            expected_node_id = semantic_by_legacy.get(mapping["sourceNodeId"])
            expected_output_id = mapping["sourceFieldId"]
        if (
            expected_node_id is None
            or mapping["targetNodeId"] != expected_node_id
            or mapping["targetOutputPortId"] != expected_output_id
        ):
            raise CompositeMigrationError(
                "compiler preview mapping does not bind the exact mapped legacy output endpoint."
            )
        matches = [
            state
            for state in preview_states
            if _object(_object(state).get("binding")).get("nodeId") == mapping["targetNodeId"]
            and _object(_object(state).get("binding")).get("outputPortId")
            == mapping["targetOutputPortId"]
        ]
        if len(matches) != 1 or _object(matches[0]).get("mediaReference") != preview_sources[source_key]:
            raise CompositeMigrationError(
                "compiler preview mapping changes or ambiguously targets a persisted media reference."
            )
        mapped_preview_keys.add(source_key)
    if mapped_preview_keys != set(preview_sources):
        missing = sorted(set(preview_sources) - mapped_preview_keys)
        raise CompositeMigrationError(
            "compiler output does not preserve every non-null legacy preview reference: "
            + ", ".join("/".join(key) for key in missing)
        )

    internal_edge_ids = {
        _required_text(edge.get("id"), "legacy Cluster internal edge.id")
        for edge in edges
        if edge.get("source") in owned_ids and edge.get("target") in owned_ids
    }
    absorbed_edge_ids = conversion.get("absorbedInternalEdgeIds")
    if (
        absorbed_edge_ids != sorted(absorbed_edge_ids)
        or len(absorbed_edge_ids) != len(set(absorbed_edge_ids))
        or set(absorbed_edge_ids) != internal_edge_ids
    ):
        raise CompositeMigrationError(
            "compiler output must explicitly receipt every absorbed internal projection edge."
        )
    migrated_edges = []
    for edge in edges:
        edge_id = _required_text(edge.get("id"), "workflow edge.id")
        source_owned = edge.get("source") in owned_ids
        target_owned = edge.get("target") in owned_ids
        if source_owned and target_owned:
            continue
        migrated = copy.deepcopy(edge)
        if source_owned:
            key = ("output", str(edge.get("source")), str(edge.get("sourceHandle") or ""))
            port_id = mapped_ports.get(key)
            if port_id is None:
                raise CompositeMigrationError(f"workflow edge {edge_id} has no exact compiler output-port mapping.")
            migrated["source"] = root_id
            migrated["sourceHandle"] = port_id
        if target_owned:
            key = ("input", str(edge.get("target")), str(edge.get("targetHandle") or ""))
            port_id = mapped_ports.get(key)
            if port_id is None:
                raise CompositeMigrationError(f"workflow edge {edge_id} has no exact compiler input-port mapping.")
            migrated["target"] = root_id
            migrated["targetHandle"] = port_id
        migrated_edges.append(migrated)

    source = instance["definitionSnapshot"]["source"]
    root_data = {
        "type": "block",
        "module": "MoDiff",
        "action": "BlockV2",
        "label": instance["definitionSnapshot"]["displayName"],
        "category": _category_for_registered_source(source),
        "params": {},
        **(
            {"description": instance["definitionSnapshot"]["description"]}
            if instance["definitionSnapshot"].get("description")
            else {}
        ),
        "resizable": True,
        "blockInstanceV2": instance,
    }
    converted_root = {
        "id": root_id,
        "type": "block",
        "position": copy.deepcopy(instance["presentation"]["position"]),
        "width": instance["presentation"]["size"]["width"],
        "height": instance["presentation"]["size"]["height"],
        **({"selected": root["selected"]} if "selected" in root else {}),
        **({"zIndex": root["zIndex"]} if "zIndex" in root else {}),
        "data": root_data,
    }
    snapshot["nodes"] = [
        converted_root if node.get("id") == root_id else node
        for node in nodes
        if node.get("id") not in child_ids
    ]
    snapshot["edges"] = migrated_edges
    converted["snapshot"] = snapshot
    return converted, {
        "instanceId": root_id,
        "definitionId": instance["definitionRef"]["definitionId"],
        "admissionId": conversion["admissionId"],
        "compiledDefinitionContentHash": conversion["compiledDefinitionContentHash"],
        "compiledDefinitionCanonicalSha256": conversion["compiledDefinitionCanonicalSha256"],
        "legacyCompositeHash": composite_hash,
        "absorbedProjectionNodeIds": sorted(child_ids),
        "absorbedInternalEdgeIds": sorted(internal_edge_ids),
        **(
            {"semanticEquivalenceAuthority": semantic_equivalence}
            if semantic_equivalence is not None
            else {}
        ),
        **(
            {"historicalCompilerMappingAuthority": historical_compiler_mapping}
            if historical_compiler_mapping is not None
            else {}
        ),
    }


def migrate_workflow_document_v1(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert every safe embedded V1 instance in one workflow document.

    Legacy Clusters remain unchanged and are counted as blocked candidates.
    Any V1 failure leaves the entire caller-owned document unchanged.
    """

    document = copy.deepcopy(_object(value))
    snapshot = _object(document.get("snapshot"))
    if not snapshot or not isinstance(snapshot.get("nodes"), list) or not isinstance(snapshot.get("edges"), list):
        raise CompositeMigrationError("saved workflow requires snapshot.nodes and snapshot.edges lists.")
    nodes = [copy.deepcopy(_object(node)) for node in snapshot["nodes"]]
    edges = [copy.deepcopy(_object(edge)) for edge in snapshot["edges"]]
    v1_ids = [
        _text(node.get("id"))
        for node in nodes
        if _object(node.get("data")).get("userBlockSnapshot") is not None
        or _text(_object(node.get("data")).get("userBlockId"))
    ]
    v1_ids = [node_id for node_id in v1_ids if node_id]
    cluster_ids = [_text(node.get("id")) for node in nodes if _is_legacy_cluster_root(node)]
    cluster_ids = [node_id for node_id in cluster_ids if node_id]
    converted_ids = []
    for root_id in v1_ids:
        root = next((node for node in nodes if node.get("id") == root_id), None)
        if root is None:
            raise CompositeMigrationError(f"legacy User Node root {root_id} disappeared during conversion.")
        converted_root, edges, removed_child_ids = _convert_v1_root(root, nodes, edges)
        nodes = [
            converted_root if node.get("id") == root_id else node
            for node in nodes
            if node.get("id") not in removed_child_ids
        ]
        converted_ids.append(root_id)
    snapshot["nodes"] = nodes
    snapshot["edges"] = edges
    document["snapshot"] = snapshot
    return document, {
        "convertedV1InstanceIds": converted_ids,
        "blockedLegacyClusterIds": sorted(cluster_ids),
    }


def _public_target(
    path: str,
    source_sha256: str,
    after_document: Any,
    *,
    kind: str,
    converted_ids: Sequence[str],
) -> dict[str, Any]:
    after_bytes = _encoded_document(after_document)
    return {
        "sourcePath": path,
        "kind": kind,
        "beforeSha256": source_sha256,
        "afterSha256": _sha256_bytes(after_bytes),
        "convertedIds": list(converted_ids),
        "byteLength": len(after_bytes),
    }


def _preview_from_sources(
    workflow_sources: Sequence[dict[str, Any]],
    reusable_sources: Sequence[dict[str, Any]],
    *,
    initial_issues: Sequence[dict[str, Any]] = (),
    compiler_supplement: Any = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    supplement_mode = compiler_supplement is not None
    supplement = _normalize_compiler_supplement(compiler_supplement)
    compiler_outputs = {
        output["sourcePath"]: output for output in supplement["compilerOutputs"]
    }
    consumed_compiler_outputs: set[tuple[str, str]] = set()
    inventory = _build_from_sources(workflow_sources, reusable_sources, initial_issues=initial_issues)
    targets = []
    writes: dict[str, bytes] = {}
    candidates = []
    blocked = []

    for source in sorted(reusable_sources, key=lambda item: item["path"]):
        if source.get("readError") or not isinstance(source.get("document"), dict):
            continue
        document = source["document"]
        if document.get("version") != 1:
            continue
        block_id = _text(document.get("id"))
        candidate = {"kind": "reusable_v1_definition", "sourcePath": source["path"], "id": block_id}
        try:
            if not block_id or not _SAFE_DEFINITION_FILE_ID.fullmatch(block_id):
                raise CompositeMigrationError(
                    "V1 definition id cannot be represented by the current /studio/blocks V2 filename contract."
                )
            if PurePosixPath(source["path"]).stem != block_id:
                raise CompositeMigrationError(
                    "V1 definition filename does not exactly match its id; migration will not rename or merge records."
                )
            converted = migrate_user_block_definition_v1(document)
            target = _public_target(
                source["path"],
                source["sourceSha256"],
                converted,
                kind="reusable_v1_definition",
                converted_ids=[block_id],
            )
            targets.append(target)
            writes[source["path"]] = _encoded_document(converted)
            candidates.append({**candidate, "status": "convertible", "targetAfterSha256": target["afterSha256"]})
        except (CompositeMigrationError, TypeError, ValueError) as exc:
            blocked_candidate = {**candidate, "status": "blocked", "reason": str(exc)}
            candidates.append(blocked_candidate)
            blocked.append(blocked_candidate)

    for source in sorted(workflow_sources, key=lambda item: item["path"]):
        if source.get("readError") or not isinstance(source.get("document"), dict):
            continue
        document = source["document"]
        try:
            converted, disposition = migrate_workflow_document_v1(document)
            cluster_ids = disposition["blockedLegacyClusterIds"]
            output = compiler_outputs.get(source["path"])
            conversions_by_id = {
                conversion["legacyInstanceId"]: conversion
                for conversion in _array(_object(output).get("conversions"))
            }
            source_hash_matches = output is None or output.get("sourceSha256") == source.get("sourceSha256")
            cluster_converted_ids = []
            for cluster_id in cluster_ids:
                if cluster_id in conversions_by_id:
                    consumed_compiler_outputs.add((source["path"], cluster_id))
                conversion = conversions_by_id.get(cluster_id) if source_hash_matches else None
                cluster_root = next(
                    (
                        _object(node)
                        for node in _array(_object(document.get("snapshot")).get("nodes"))
                        if _object(node).get("id") == cluster_id
                    ),
                    {},
                )
                semantic_equivalence = _available_semantic_equivalence_authority(cluster_root)
                historical_compiler_mapping = (
                    _available_historical_compiler_mapping_authority(cluster_root)
                    if semantic_equivalence is None
                    else None
                )
                candidate = {
                    "kind": "legacy_registered_cluster_instance",
                    "sourcePath": source["path"],
                    "id": cluster_id,
                    # These hashes are non-secret compiler inputs. Expose them
                    # in the default read-only preview so the visible client
                    # can bind an exact compiler output to the backend-owned
                    # source bytes without attempting to reproduce the file's
                    # byte-level JSON encoding in the browser.
                    "sourceSha256": source.get("sourceSha256"),
                    "legacyCompositeHash": legacy_registered_cluster_composite_hash(
                        document, cluster_id
                    ),
                    **(
                        {"semanticEquivalenceAuthority": semantic_equivalence}
                        if semantic_equivalence is not None
                        else {}
                    ),
                    **(
                        {
                            "historicalCompilerMappingAuthority": historical_compiler_mapping
                        }
                        if historical_compiler_mapping is not None
                        else {}
                    ),
                }
                if conversion is None:
                    reason = (
                        "Compiler supplement sourceSha256 does not match the reviewed workflow source."
                        if output is not None and not source_hash_matches
                        else (
                            (
                                "A checked-in historical semantic-equivalence receipt is available; "
                                "requires an exact compiler output that references it."
                            )
                            if semantic_equivalence is not None
                            else (
                                "A checked-in archived compiler mapping is available; requires an exact compiler output that references it."
                                if historical_compiler_mapping is not None
                                else (
                                    "Requires the exact pinned registered-catalog BlockDefinitionV2 compiler output and, for a historical manifest, an exact reviewed compiler mapping or semantic-equivalence receipt; presentation children are not execution authority."
                                )
                            )
                        )
                    )
                    blocked_candidate = {**candidate, "status": "blocked", "reason": reason}
                    candidates.append(blocked_candidate)
                    blocked.append(blocked_candidate)
                    continue
                try:
                    converted, receipt = _convert_registered_cluster_root(
                        converted,
                        conversion,
                        legacy_source_document=document,
                    )
                    cluster_converted_ids.append(cluster_id)
                    candidates.append(
                        {
                            **candidate,
                            "status": "convertible",
                            "compilerReceipt": receipt,
                        }
                    )
                except (CompositeMigrationError, TypeError, ValueError) as exc:
                    blocked_candidate = {**candidate, "status": "blocked", "reason": str(exc)}
                    candidates.append(blocked_candidate)
                    blocked.append(blocked_candidate)
            for conversion_id in sorted(set(conversions_by_id) - set(cluster_ids)):
                consumed_compiler_outputs.add((source["path"], conversion_id))
                blocked_candidate = {
                    "kind": "registered_cluster_compiler_output",
                    "sourcePath": source["path"],
                    "sourceSha256": source.get("sourceSha256"),
                    "id": conversion_id,
                    "status": "blocked",
                    "reason": "Compiler output does not identify a legacy registered Cluster in this workflow.",
                }
                candidates.append(blocked_candidate)
                blocked.append(blocked_candidate)
            converted_ids = [*disposition["convertedV1InstanceIds"], *cluster_converted_ids]
            if not converted_ids:
                continue
            target = _public_target(
                source["path"],
                source["sourceSha256"],
                converted,
                kind=("workflow_composite_instances" if cluster_converted_ids else "workflow_v1_instances"),
                converted_ids=converted_ids,
            )
            targets.append(target)
            writes[source["path"]] = _encoded_document(converted)
            for candidate in candidates:
                if candidate.get("sourcePath") == source["path"] and candidate.get("status") == "convertible":
                    candidate["targetAfterSha256"] = target["afterSha256"]
            candidates.extend(
                {
                    "kind": "workflow_v1_instance",
                    "sourcePath": source["path"],
                    "id": root_id,
                    "status": "convertible",
                    "targetAfterSha256": target["afterSha256"],
                }
                for root_id in disposition["convertedV1InstanceIds"]
            )
        except (CompositeMigrationError, TypeError, ValueError) as exc:
            candidate = {
                "kind": "workflow_v1_instances",
                "sourcePath": source["path"],
                "id": None,
                "status": "blocked",
                "reason": str(exc),
            }
            candidates.append(candidate)
            blocked.append(candidate)

    known_workflow_paths = {source["path"] for source in workflow_sources}
    for output in supplement["compilerOutputs"]:
        for conversion in output["conversions"]:
            key = (output["sourcePath"], conversion["legacyInstanceId"])
            if key in consumed_compiler_outputs:
                continue
            reason = (
                "Compiler output sourcePath does not exist in the current workflow store."
                if output["sourcePath"] not in known_workflow_paths
                else "Compiler output source could not be read and remains unapplied."
            )
            blocked_candidate = {
                "kind": "registered_cluster_compiler_output",
                "sourcePath": output["sourcePath"],
                "sourceSha256": output["sourceSha256"],
                "id": conversion["legacyInstanceId"],
                "status": "blocked",
                "reason": reason,
            }
            candidates.append(blocked_candidate)
            blocked.append(blocked_candidate)

    targets.sort(key=lambda target: target["sourcePath"])
    candidates.sort(key=lambda item: (item["sourcePath"], item["kind"], str(item.get("id") or "")))
    blocked.sort(key=lambda item: (item["sourcePath"], item["kind"], str(item.get("id") or "")))
    source_set = [
        {"sourcePath": source["path"], "sourceSha256": source.get("sourceSha256")}
        for source in sorted([*workflow_sources, *reusable_sources], key=lambda item: item["path"])
    ]
    preview = {
        "schemaVersion": MIGRATION_SCHEMA_VERSION,
        "kind": "legacy_composite_to_block_v2_migration",
        "mode": "read_only_preview",
        "boundary": {
            "writesFiles": False,
            "requiresExplicitApplyAuthority": True,
            "createsExactBackupsBeforeReplacement": True,
            "deletesRecords": False,
            "mergesRecords": False,
            "legacyClusterConversionRequiresRegisteredCompiler": True,
            **(
                {"compilerSupplementRequiredForRegisteredClusters": True}
                if supplement_mode
                else {}
            ),
            "containsPromptAndParameterValues": False,
        },
        **(
            {
                "compilerSupplement": {
                    "provided": bool(supplement["compilerOutputs"]),
                    "contentHash": _sha256_json(supplement) if supplement["compilerOutputs"] else None,
                    "sourceCount": len(supplement["compilerOutputs"]),
                    "conversionCount": sum(
                        len(output["conversions"]) for output in supplement["compilerOutputs"]
                    ),
                }
            }
            if supplement_mode
            else {}
        ),
        "inventoryReportHash": inventory["reportHash"],
        "sourceSetHash": _sha256_json(source_set),
        "summary": {
            "sourceCount": len(source_set),
            "targetFileCount": len(targets),
            "convertibleCandidateCount": sum(item["status"] == "convertible" for item in candidates),
            "blockedCandidateCount": len(blocked),
            "legacyClusterBlockedCount": sum(item["kind"] == "legacy_registered_cluster_instance" for item in blocked),
            **(
                {
                    "registeredClusterConvertibleCount": sum(
                        item["kind"] == "legacy_registered_cluster_instance"
                        and item["status"] == "convertible"
                        for item in candidates
                    )
                }
                if supplement_mode
                else {}
            ),
            "hasChanges": bool(targets),
        },
        "targets": targets,
        "candidates": candidates,
        "blocked": blocked,
        "inventoryIssues": copy.deepcopy(inventory["issues"]),
    }
    identity = _sha256_json(preview).removeprefix("sha256:")[:24]
    preview["migrationId"] = f"block-v2-migration-{identity}"
    preview["planHash"] = _sha256_json(preview)
    return preview, writes


def build_composite_migration_preview(
    workflow_documents: Mapping[str, Any],
    reusable_block_documents: Mapping[str, Any],
    *,
    compiler_supplement: Any = None,
) -> dict[str, Any]:
    """Build a deterministic preview from caller-owned values without writes."""

    workflow_sources = [
        {
            "path": path,
            "sourceSha256": _sha256_json(document),
            "document": copy.deepcopy(document),
            "readError": None,
        }
        for path, document in workflow_documents.items()
    ]
    reusable_sources = [
        {
            "path": path,
            "sourceSha256": _sha256_json(document),
            "document": copy.deepcopy(document),
            "readError": None,
        }
        for path, document in reusable_block_documents.items()
    ]
    preview, _ = _preview_from_sources(
        workflow_sources,
        reusable_sources,
        compiler_supplement=compiler_supplement,
    )
    return preview


def _scan_plan(
    data_dir: Path, *, compiler_supplement: Any = None
) -> tuple[dict[str, Any], dict[str, bytes]]:
    workflow_sources, workflow_issues = _load_sources(data_dir, Path("user-workflows"))
    reusable_sources, reusable_issues = _load_sources(data_dir, Path("studio/blocks"))
    return _preview_from_sources(
        workflow_sources,
        reusable_sources,
        initial_issues=[*workflow_issues, *reusable_issues],
        compiler_supplement=compiler_supplement,
    )


def scan_composite_migration_preview(
    data_dir: str | Path, *, compiler_supplement: Any = None
) -> dict[str, Any]:
    """Return the current deterministic plan. This function never writes."""

    preview, _ = _scan_plan(Path(data_dir), compiler_supplement=compiler_supplement)
    return preview


def _migration_root(data_dir: Path) -> Path:
    root = data_dir / "studio" / "composite-migrations"
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise CompositeMigrationConflictError("Composite migration journal directory is unsafe.")
    return root


def _safe_relative_target(data_dir: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    allowed = (len(path.parts) == 2 and path.parts[0] == "user-workflows") or (
        len(path.parts) == 3 and path.parts[:2] == ("studio", "blocks")
    )
    if not allowed or path.suffix != ".json" or any(part in {"", ".", ".."} for part in path.parts):
        raise CompositeMigrationConflictError("Migration target is outside the two reviewed JSON stores.")
    target = data_dir.joinpath(*path.parts)
    if target.is_symlink() or target.parent.is_symlink():
        raise CompositeMigrationConflictError(f"Migration target {relative} is an unsafe symlink.")
    try:
        target.resolve(strict=False).relative_to(data_dir.resolve(strict=False))
    except ValueError as exc:
        raise CompositeMigrationConflictError(f"Migration target {relative} escapes the data directory.") from exc
    return target


def _fsync_directory(path: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _write_new_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.migration-{uuid.uuid4().hex}.tmp"
    try:
        with open(temporary, "xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_path(data_dir: Path, migration_id: str) -> Path:
    if not _MIGRATION_ID.fullmatch(migration_id):
        raise CompositeMigrationError("Migration id is malformed.")
    return _migration_root(data_dir) / migration_id / "manifest.json"


def _read_manifest(data_dir: Path, migration_id: str) -> dict[str, Any] | None:
    path = _manifest_path(data_dir, migration_id)
    if not path.is_file():
        return None
    if path.is_symlink() or path.parent.is_symlink():
        raise CompositeMigrationConflictError("Migration journal is an unsafe symlink.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("migrationId") != migration_id:
        raise CompositeMigrationConflictError("Migration journal is malformed.")
    return value


def _write_manifest(data_dir: Path, manifest: Mapping[str, Any]) -> None:
    path = _manifest_path(data_dir, _required_text(manifest.get("migrationId"), "manifest migrationId"))
    _atomic_write(path, _encoded_document(manifest))


def _current_target_hash(data_dir: Path, relative: str) -> str | None:
    target = _safe_relative_target(data_dir, relative)
    return _sha256_bytes(target.read_bytes()) if target.is_file() else None


def _journal_status(data_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    targets = []
    for target in _array(manifest.get("targets")):
        item = _object(target)
        relative = _required_text(item.get("sourcePath"), "manifest target sourcePath")
        current = _current_target_hash(data_dir, relative)
        if current == item.get("afterSha256"):
            state = "applied"
        elif current == item.get("beforeSha256"):
            state = "source"
        elif current is None:
            state = "missing"
        else:
            state = "conflict"
        targets.append({"sourcePath": relative, "currentSha256": current, "state": state})
    states = {target["state"] for target in targets}
    if states <= {"applied"}:
        effective = "applied"
    elif states <= {"source"}:
        effective = "rolled_back"
    elif states <= {"source", "applied"}:
        effective = "interrupted"
    else:
        effective = "conflict"
    return {
        "migrationId": manifest.get("migrationId"),
        "journalState": manifest.get("state"),
        "effectiveState": effective,
        "planHash": manifest.get("planHash"),
        "inventoryReportHash": manifest.get("inventoryReportHash"),
        "targets": targets,
        "rollbackAvailable": effective in {"applied", "interrupted", "rolled_back"},
        "resumeAvailable": False,
    }


def list_composite_migrations(data_dir: str | Path) -> dict[str, Any]:
    root = _migration_root(Path(data_dir))
    if not root.exists():
        return {"schemaVersion": 1, "migrations": []}
    result = []
    for directory in sorted(root.iterdir(), key=lambda item: item.name):
        if not directory.is_dir() or directory.is_symlink() or not _MIGRATION_ID.fullmatch(directory.name):
            continue
        manifest = _read_manifest(Path(data_dir), directory.name)
        if manifest:
            result.append(_journal_status(Path(data_dir), manifest))
    return {"schemaVersion": 1, "migrations": result}


def composite_migration_status(data_dir: str | Path, migration_id: str) -> dict[str, Any] | None:
    manifest = _read_manifest(Path(data_dir), migration_id)
    return _journal_status(Path(data_dir), manifest) if manifest else None


def _prepare_journal(data_dir: Path, preview: Mapping[str, Any]) -> dict[str, Any]:
    migration_id = _required_text(preview.get("migrationId"), "migrationId")
    migration_dir = _manifest_path(data_dir, migration_id).parent
    if migration_dir.exists():
        raise CompositeMigrationConflictError("Migration journal already exists and requires status inspection.")
    temporary = migration_dir.parent / f".{migration_id}-{uuid.uuid4().hex}.tmp"
    try:
        backup_root = temporary / "backups"
        targets = []
        for target in _array(preview.get("targets")):
            item = _object(target)
            relative = _required_text(item.get("sourcePath"), "preview target sourcePath")
            source = _safe_relative_target(data_dir, relative)
            raw = source.read_bytes()
            if _sha256_bytes(raw) != item.get("beforeSha256"):
                raise CompositeMigrationConflictError(
                    f"Migration source {relative} changed after preview; refresh the preview."
                )
            backup = backup_root.joinpath(*PurePosixPath(relative).parts)
            _write_new_file(backup, raw)
            targets.append(copy.deepcopy(item))
        manifest = {
            "schemaVersion": MIGRATION_SCHEMA_VERSION,
            "kind": "legacy_composite_to_block_v2_transaction",
            "migrationId": migration_id,
            "planHash": preview.get("planHash"),
            "inventoryReportHash": preview.get("inventoryReportHash"),
            "sourceSetHash": preview.get("sourceSetHash"),
            "compilerSupplement": copy.deepcopy(preview.get("compilerSupplement")),
            "compilerReceipts": [
                copy.deepcopy(candidate["compilerReceipt"])
                for candidate in _array(preview.get("candidates"))
                if isinstance(candidate, dict) and candidate.get("compilerReceipt") is not None
            ],
            "state": "prepared",
            "targets": targets,
        }
        _write_new_file(temporary / "manifest.json", _encoded_document(manifest))
        migration_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, migration_dir)
        _fsync_directory(migration_dir.parent)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _backup_bytes(data_dir: Path, migration_id: str, relative: str, expected_hash: str) -> bytes:
    journal = _manifest_path(data_dir, migration_id).parent
    backup = journal / "backups" / Path(*PurePosixPath(relative).parts)
    try:
        backup.resolve(strict=False).relative_to(journal.resolve(strict=False))
    except ValueError as exc:
        raise CompositeMigrationConflictError(f"Exact backup for {relative} escapes its journal.") from exc
    if not backup.is_file() or backup.is_symlink():
        raise CompositeMigrationConflictError(f"Exact backup for {relative} is missing or unsafe.")
    raw = backup.read_bytes()
    if _sha256_bytes(raw) != expected_hash:
        raise CompositeMigrationConflictError(f"Exact backup for {relative} failed its hash check.")
    return raw


def apply_composite_migration(
    data_dir: str | Path,
    *,
    migration_id: str,
    plan_hash: str,
    confirmation: str,
    allow_blocked_candidates: bool = False,
    compiler_supplement: Any = None,
) -> dict[str, Any]:
    """Apply the exact current preview after explicit operator authorization."""

    if confirmation != APPLY_CONFIRMATION:
        raise CompositeMigrationAuthorizationError(f"confirmation must equal {APPLY_CONFIRMATION}.")
    if not _MIGRATION_ID.fullmatch(str(migration_id or "")) or not _text(plan_hash):
        raise CompositeMigrationAuthorizationError("migrationId and planHash are required from the exact preview.")
    supplement_mode = compiler_supplement is not None
    normalized_supplement = _normalize_compiler_supplement(compiler_supplement)
    supplement_content_hash = (
        _sha256_json(normalized_supplement) if normalized_supplement["compilerOutputs"] else None
    )
    root = Path(data_dir)
    with STUDIO_PERSISTENCE_LOCK:
        existing = _read_manifest(root, migration_id)
        if existing:
            reviewed_supplement = _object(existing.get("compilerSupplement"))
            if reviewed_supplement.get("provided") and reviewed_supplement.get("contentHash") != supplement_content_hash:
                raise CompositeMigrationConflictError(
                    "Migration apply must resend the exact reviewed compiler supplement."
                )
            status = _journal_status(root, existing)
            if status["effectiveState"] == "applied" and existing.get("planHash") == plan_hash:
                return {"error": False, "idempotent": True, "status": status}
            if status["effectiveState"] != "rolled_back":
                raise CompositeMigrationConflictError(
                    "Migration journal requires explicit rollback/recovery before another apply."
                )

        preview, writes = _scan_plan(
            root,
            compiler_supplement=normalized_supplement if supplement_mode else None,
        )
        if preview["migrationId"] != migration_id or preview["planHash"] != plan_hash:
            raise CompositeMigrationConflictError("Migration sources changed after preview; request a fresh preview.")
        if not preview["summary"]["hasChanges"]:
            raise CompositeMigrationError("Migration preview contains no convertible targets.")
        if preview["summary"]["blockedCandidateCount"] and not allow_blocked_candidates:
            raise CompositeMigrationAuthorizationError(
                "Preview contains blocked candidates; set allowBlockedCandidates=true to apply only the listed safe targets."
            )

        if existing:
            manifest = existing
            for target in manifest["targets"]:
                _backup_bytes(root, migration_id, target["sourcePath"], target["beforeSha256"])
        else:
            manifest = _prepare_journal(root, preview)
        manifest["state"] = "applying"
        _write_manifest(root, manifest)
        changed = []
        try:
            for target in manifest["targets"]:
                relative = target["sourcePath"]
                current = _current_target_hash(root, relative)
                if current == target["afterSha256"]:
                    continue
                if current != target["beforeSha256"]:
                    raise CompositeMigrationConflictError(
                        f"Migration target {relative} changed before replacement; rollback is required."
                    )
                payload = writes.get(relative)
                if payload is None or _sha256_bytes(payload) != target["afterSha256"]:
                    raise CompositeMigrationConflictError(
                        f"Migration output for {relative} no longer matches the plan."
                    )
                _atomic_write(_safe_relative_target(root, relative), payload)
                changed.append(relative)
        except Exception:
            rollback_conflicts = []
            for target in reversed(manifest["targets"]):
                relative = target["sourcePath"]
                current = _current_target_hash(root, relative)
                if current == target["beforeSha256"]:
                    continue
                if current != target["afterSha256"]:
                    rollback_conflicts.append(relative)
                    continue
                backup = _backup_bytes(root, migration_id, relative, target["beforeSha256"])
                _atomic_write(_safe_relative_target(root, relative), backup)
            manifest["state"] = "recovery_required" if rollback_conflicts else "rolled_back"
            _write_manifest(root, manifest)
            if rollback_conflicts:
                raise CompositeMigrationConflictError(
                    "Apply failed and automatic rollback found independently modified targets: "
                    + ", ".join(sorted(rollback_conflicts))
                )
            raise
        manifest["state"] = "applied"
        _write_manifest(root, manifest)
        return {
            "error": False,
            "idempotent": False,
            "changedPaths": changed,
            "status": _journal_status(root, manifest),
        }


def rollback_composite_migration(
    data_dir: str | Path,
    *,
    migration_id: str,
    confirmation: str,
) -> dict[str, Any]:
    """Restore every exact backup, refusing unknown post-migration edits."""

    if confirmation != ROLLBACK_CONFIRMATION:
        raise CompositeMigrationAuthorizationError(f"confirmation must equal {ROLLBACK_CONFIRMATION}.")
    root = Path(data_dir)
    with STUDIO_PERSISTENCE_LOCK:
        manifest = _read_manifest(root, migration_id)
        if not manifest:
            raise CompositeMigrationError("Migration journal was not found.")
        status = _journal_status(root, manifest)
        conflicts = [
            target["sourcePath"] for target in status["targets"] if target["state"] in {"conflict", "missing"}
        ]
        if conflicts:
            raise CompositeMigrationConflictError(
                "Rollback will not overwrite independently changed or missing targets: " + ", ".join(conflicts)
            )
        # Verify the complete recovery set before changing the first target.
        # A missing/tampered backup must not create a partially rolled-back
        # workspace merely because its target sorts later in the transaction.
        for target in manifest["targets"]:
            _backup_bytes(root, migration_id, target["sourcePath"], target["beforeSha256"])
        restored = []
        manifest["state"] = "rolling_back"
        _write_manifest(root, manifest)
        try:
            for target in reversed(manifest["targets"]):
                relative = target["sourcePath"]
                current = _current_target_hash(root, relative)
                if current == target["beforeSha256"]:
                    continue
                if current != target["afterSha256"]:
                    raise CompositeMigrationConflictError(f"Rollback target {relative} changed during recovery.")
                backup = _backup_bytes(root, migration_id, relative, target["beforeSha256"])
                _atomic_write(_safe_relative_target(root, relative), backup)
                restored.append(relative)
        except Exception:
            manifest["state"] = "recovery_required"
            _write_manifest(root, manifest)
            raise
        manifest["state"] = "rolled_back"
        _write_manifest(root, manifest)
        return {
            "error": False,
            "idempotent": not restored,
            "restoredPaths": restored,
            "status": _journal_status(root, manifest),
        }
