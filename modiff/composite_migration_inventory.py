"""Deterministic, read-only inventory for legacy composite migration planning.

This module deliberately performs no conversion and exposes no write helper.
It inventories backend-saved workflow JSON and reusable User Node records so a
human can review identity, value, interface, topology, and ambiguity before a
future migration is authorized.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from modiff.block_definition_v2 import validate_block_definition_v2


REPORT_SCHEMA_VERSION = 1
MAX_SOURCE_FILES = 10_000
MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_SOURCE_BYTES = 256 * 1024 * 1024


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _array(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _document_sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _report_hash(report: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(report).encode("utf-8")).hexdigest()


def _source_record(path: str, document: Any, source_sha256: str | None = None) -> dict[str, Any]:
    return {
        "path": path,
        "sourceSha256": source_sha256 or _document_sha256(document),
        "document": copy.deepcopy(document),
        "readError": None,
    }


def _error_source_record(path: str, *, source_sha256: str | None, code: str, message: str) -> dict[str, Any]:
    return {
        "path": path,
        "sourceSha256": source_sha256,
        "document": None,
        "readError": {"code": code, "message": message},
    }


class _Issues:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(
        self,
        source_path: str,
        path: str,
        code: str,
        message: str,
        *,
        severity: str = "error",
    ) -> str:
        self.items.append(
            {
                "sourcePath": source_path,
                "path": path,
                "code": code,
                "severity": severity,
                "message": message,
            }
        )
        return code

    def sorted(self) -> list[dict[str, Any]]:
        return sorted(
            self.items,
            key=lambda item: (
                item["sourcePath"],
                item["path"],
                item["severity"],
                item["code"],
                item["message"],
            ),
        )


def _position_and_size(node: dict[str, Any], issues: _Issues, source_path: str, node_path: str) -> dict[str, Any]:
    position = copy.deepcopy(node.get("position"))
    if not (isinstance(position, dict) and _finite_number(position.get("x")) and _finite_number(position.get("y"))):
        issues.add(
            source_path, f"{node_path}.position", "node_position_invalid", "Node position is missing or non-finite."
        )
    width = node.get("width")
    height = node.get("height")
    if width is not None and (not _finite_number(width) or width <= 0):
        issues.add(
            source_path, f"{node_path}.width", "node_width_invalid", "Node width must be a positive finite number."
        )
    if height is not None and (not _finite_number(height) or height <= 0):
        issues.add(
            source_path, f"{node_path}.height", "node_height_invalid", "Node height must be a positive finite number."
        )
    return {
        "position": position,
        "size": {"width": copy.deepcopy(width), "height": copy.deepcopy(height)},
    }


def _parameter_inventory(params_value: Any) -> list[dict[str, Any]]:
    params = _object(params_value)
    result = []
    for param_id in sorted(params):
        param = _object(params[param_id])
        entry: dict[str, Any] = {
            "paramId": param_id,
            "label": copy.deepcopy(param.get("label")),
            "type": copy.deepcopy(param.get("type")),
            "display": copy.deepcopy(param.get("display")),
            "valuePresent": "value" in param,
        }
        if "value" in param:
            entry["value"] = copy.deepcopy(param["value"])
        result.append(entry)
    return result


def _prompt_inventory(parameters: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "paramId": parameter["paramId"],
            "valuePresent": parameter["valuePresent"],
            **({"value": copy.deepcopy(parameter["value"])} if "value" in parameter else {}),
        }
        for parameter in parameters
        if "prompt" in parameter["paramId"].lower()
    ]


def _graph_parameter_inventory(nodes_value: Any) -> list[dict[str, Any]]:
    result = []
    for index, node_value in enumerate(_array(nodes_value)):
        node = _object(node_value)
        data = _object(node.get("data"))
        parameters = _parameter_inventory(data.get("params"))
        if not parameters:
            continue
        result.append(
            {
                "nodeId": _text(node.get("id")) or _text(node.get("nodeId")) or f"index:{index}",
                "parameters": parameters,
                "prompts": _prompt_inventory(parameters),
            }
        )
    return result


def _projected_ports(params_value: Any) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {"inputs": [], "outputs": []}
    for param_id, param_value in sorted(_object(params_value).items()):
        param = _object(param_value)
        field_options = _object(param.get("fieldOptions"))
        direction = field_options.get("huggingFaceClusterPortDirection") or param.get("display")
        if direction not in {"input", "output"}:
            continue
        result[f"{direction}s"].append(
            {
                "portId": param_id,
                "label": copy.deepcopy(param.get("label")),
                "valueType": copy.deepcopy(param.get("type")),
                "binding": {
                    "nodeId": copy.deepcopy(field_options.get("huggingFaceClusterPortNodeId")),
                    "fieldOrPortId": copy.deepcopy(field_options.get("huggingFaceClusterPortField")),
                },
                "source": "root_params",
            }
        )
    return result


def _declared_v1_ports(snapshot: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {"inputs": [], "outputs": []}
    for direction in ("inputs", "outputs"):
        for port_value in _array(snapshot.get(direction)):
            port = _object(port_value)
            result[direction].append(
                {
                    "portId": copy.deepcopy(port.get("id")),
                    "label": copy.deepcopy(port.get("label")),
                    "valueType": copy.deepcopy(port.get("type")),
                    "binding": {
                        "nodeId": copy.deepcopy(port.get("nodeId")),
                        "fieldOrPortId": copy.deepcopy(port.get("paramKey")),
                    },
                    "source": "userBlockSnapshot",
                }
            )
    return result


def _declared_v2_ports(instance: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    snapshot = _object(instance.get("definitionSnapshot"))
    boundary = _object(snapshot.get("boundary"))
    result: dict[str, list[dict[str, Any]]] = {"inputs": [], "outputs": []}
    for direction in ("inputs", "outputs"):
        for port_value in _array(boundary.get(direction)):
            port = _object(port_value)
            result[direction].append(
                {
                    "portId": copy.deepcopy(port.get("portId")),
                    "label": copy.deepcopy(port.get("label")),
                    "valueType": copy.deepcopy(port.get("valueType")),
                    "required": copy.deepcopy(port.get("required")),
                    "multiple": copy.deepcopy(port.get("multiple")),
                    "binding": copy.deepcopy(port.get("binding")),
                    "source": "BlockInstanceV2.definitionSnapshot.boundary",
                }
            )
    return result


def _legacy_cluster_ref(data: dict[str, Any]) -> dict[str, Any]:
    instance = _object(data.get("huggingFaceClusterInstance"))
    execution = _object(instance.get("execution"))
    return {
        "instanceId": copy.deepcopy(instance.get("instanceId")),
        "definition": copy.deepcopy(instance.get("definition")),
        "admissionId": copy.deepcopy(execution.get("admissionId")),
        # This is an immutable execution-identity receipt, not a prompt or
        # parameter value.  The recovery audit needs all three fields to tell
        # a complete historical tuple from a manifest reference whose
        # execution authority was lost.  Keeping it beside the admission also
        # avoids treating a matching admission label as sufficient authority.
        "studioExecutionSpec": copy.deepcopy(execution.get("studioExecutionSpec")),
    }


def _legacy_user_ref(data: dict[str, Any]) -> dict[str, Any]:
    snapshot = _object(data.get("userBlockSnapshot"))
    return {
        "userBlockId": copy.deepcopy(data.get("userBlockId") or snapshot.get("id")),
        "snapshotId": copy.deepcopy(snapshot.get("id")),
        "snapshotVersion": copy.deepcopy(snapshot.get("version")),
        "origin": copy.deepcopy(snapshot.get("origin")),
    }


def _v2_ref(data: dict[str, Any]) -> dict[str, Any]:
    instance = _object(data.get("blockInstanceV2"))
    snapshot = _object(instance.get("definitionSnapshot"))
    return {
        "instanceId": copy.deepcopy(instance.get("instanceId")),
        "definitionRef": copy.deepcopy(instance.get("definitionRef")),
        "source": copy.deepcopy(snapshot.get("source")),
        "customization": copy.deepcopy(instance.get("customization")),
        "effectiveGraphHash": copy.deepcopy(_object(instance.get("effectiveGraph")).get("graphHash")),
    }


def _root_families(node: dict[str, Any]) -> list[str]:
    data = _object(node.get("data"))
    families = []
    if "blockInstanceV2" in data:
        families.append("block_v2")
    if "userBlockSnapshot" in data or _text(data.get("userBlockId")):
        families.append("legacy_user_block_v1")
    if (
        "huggingFaceClusterInstance" in data
        or data.get("huggingFaceClusterRole") == "root"
        or node.get("type") == "cluster"
        or data.get("type") == "cluster"
    ):
        families.append("legacy_cluster")
    return families


def _derived_owners(node: dict[str, Any]) -> list[tuple[str, str]]:
    data = _object(node.get("data"))
    result = []
    for kind, field in (
        ("legacy_cluster_child", "huggingFaceClusterInstanceId"),
        ("legacy_user_block_child", "userBlockInstanceId"),
        ("block_v2_projection", "blockProjectionOwnerId"),
    ):
        owner = _text(data.get(field))
        if owner:
            result.append((kind, owner))
    return result


def _root_record(
    node: dict[str, Any],
    node_index: int,
    source_path: str,
    issues: _Issues,
) -> tuple[dict[str, Any], list[str]]:
    node_path = f"snapshot.nodes[{node_index}]"
    data = _object(node.get("data"))
    families = _root_families(node)
    issue_codes = []
    if len(families) > 1:
        issue_codes.append(
            issues.add(
                source_path,
                node_path,
                "composite_authority_ambiguous",
                "Node contains markers for more than one composite authority.",
            )
        )
        classification = "ambiguous_composite_root"
    else:
        classification = {
            "legacy_cluster": "legacy_cluster_root",
            "legacy_user_block_v1": "legacy_user_block_v1_root",
            "block_v2": "block_v2_root",
        }[families[0]]
    node_id = _text(node.get("id"))
    if not node_id:
        issue_codes.append(
            issues.add(source_path, f"{node_path}.id", "composite_node_id_missing", "Composite root has no node ID.")
        )
    parameters = _parameter_inventory(data.get("params"))
    cluster_instance = _object(data.get("huggingFaceClusterInstance"))
    user_snapshot = _object(data.get("userBlockSnapshot"))
    v2_instance = _object(data.get("blockInstanceV2"))
    if "legacy_cluster" in families:
        if not cluster_instance:
            issue_codes.append(
                issues.add(
                    source_path,
                    f"{node_path}.data.huggingFaceClusterInstance",
                    "legacy_cluster_instance_missing",
                    "Legacy Cluster root has no embedded instance authority.",
                )
            )
        elif node_id and cluster_instance.get("instanceId") != node_id:
            issue_codes.append(
                issues.add(
                    source_path,
                    f"{node_path}.data.huggingFaceClusterInstance.instanceId",
                    "legacy_cluster_instance_id_mismatch",
                    "Legacy Cluster instance ID does not match its canvas root ID.",
                )
            )
    if "legacy_user_block_v1" in families and not user_snapshot:
        issue_codes.append(
            issues.add(
                source_path,
                f"{node_path}.data.userBlockSnapshot",
                "legacy_user_snapshot_missing",
                "Legacy User Node references a definition but has no embedded snapshot.",
            )
        )
    if "block_v2" in families:
        if not v2_instance:
            issue_codes.append(
                issues.add(
                    source_path,
                    f"{node_path}.data.blockInstanceV2",
                    "block_v2_instance_malformed",
                    "Block V2 root has no object instance payload.",
                )
            )
        elif node_id and v2_instance.get("instanceId") != node_id:
            issue_codes.append(
                issues.add(
                    source_path,
                    f"{node_path}.data.blockInstanceV2.instanceId",
                    "block_v2_instance_id_mismatch",
                    "Block V2 instance ID does not match its canvas root ID.",
                )
            )
    if "legacy_user_block_v1" in families:
        ports = _declared_v1_ports(user_snapshot)
        graph_parameters = _graph_parameter_inventory(user_snapshot.get("nodes"))
    elif "block_v2" in families:
        ports = _declared_v2_ports(v2_instance)
        graph_parameters = _graph_parameter_inventory(_object(v2_instance.get("effectiveGraph")).get("nodes"))
    else:
        ports = _projected_ports(data.get("params"))
        graph_parameters = []
    presentation = {
        "rootUiState": copy.deepcopy(data.get("uiState")),
        "clusterPresentation": copy.deepcopy(cluster_instance.get("presentation")),
        "blockV2Presentation": copy.deepcopy(v2_instance.get("presentation")),
    }
    instance_values = {
        "clusterParameterOverrides": copy.deepcopy(cluster_instance.get("parameterOverrides")),
        "clusterExecutionParameterOverrides": copy.deepcopy(
            _object(cluster_instance.get("execution")).get("parameterOverrides")
        ),
        "blockV2Values": copy.deepcopy(v2_instance.get("values")),
    }
    record = {
        "classification": classification,
        "authorityMarkers": families,
        "nodeId": copy.deepcopy(node.get("id")),
        "reactFlowType": copy.deepcopy(node.get("type")),
        "dataType": copy.deepcopy(data.get("type")),
        "parentId": copy.deepcopy(node.get("parentId")),
        "canvas": _position_and_size(node, issues, source_path, node_path),
        "presentation": presentation,
        "sourceRefs": {
            "legacyCluster": _legacy_cluster_ref(data) if "legacy_cluster" in families else None,
            "legacyUserBlock": _legacy_user_ref(data) if "legacy_user_block_v1" in families else None,
            "blockV2": _v2_ref(data) if "block_v2" in families else None,
        },
        "parameters": parameters,
        "prompts": _prompt_inventory(parameters),
        "graphParameters": graph_parameters,
        "instanceValues": instance_values,
        "ports": ports,
        "derivedChildIds": [],
        "externalEdges": [],
        "definitionResolution": None,
        "issueCodes": sorted(set(issue_codes)),
        "migrationDisposition": "inventory_only_no_conversion",
    }
    return record, families


def _derived_record(
    node: dict[str, Any],
    node_index: int,
    kind: str,
    owner_id: str,
    source_path: str,
    issues: _Issues,
) -> dict[str, Any]:
    data = _object(node.get("data"))
    parameters = _parameter_inventory(data.get("params"))
    return {
        "classification": kind,
        "nodeId": copy.deepcopy(node.get("id")),
        "ownerId": owner_id,
        "role": copy.deepcopy(data.get("huggingFaceClusterRole") or data.get("blockProjectionKind")),
        "semanticId": copy.deepcopy(
            data.get("huggingFaceClusterSemanticId")
            or data.get("userBlockSourceNodeId")
            or data.get("blockProjectionNodeId")
        ),
        "canvas": _position_and_size(node, issues, source_path, f"snapshot.nodes[{node_index}]"),
        "parameters": parameters,
        "prompts": _prompt_inventory(parameters),
    }


def _edge_inventory(edge: dict[str, Any], direction: str, owned_endpoint: str) -> dict[str, Any]:
    return {
        "edgeId": copy.deepcopy(edge.get("id")),
        "direction": direction,
        "sourceNodeId": copy.deepcopy(edge.get("source")),
        "sourcePortId": copy.deepcopy(edge.get("sourceHandle")),
        "targetNodeId": copy.deepcopy(edge.get("target")),
        "targetPortId": copy.deepcopy(edge.get("targetHandle")),
        "ownedEndpointNodeId": owned_endpoint,
    }


def _reusable_record(
    source: dict[str, Any],
    issues: _Issues,
) -> dict[str, Any]:
    source_path = source["path"]
    if source["readError"]:
        error = source["readError"]
        issues.add(source_path, "$", error["code"], error["message"])
        return {
            "sourcePath": source_path,
            "sourceSha256": source["sourceSha256"],
            "status": "error",
            "classification": "unreadable_reusable_block_record",
            "blockId": None,
            "name": None,
            "sourceRefs": None,
            "ports": {"inputs": [], "outputs": []},
            "graph": {"nodeIds": [], "edgeIds": [], "nodeCount": 0, "edgeCount": 0},
            "parameters": [],
            "prompts": [],
            "issueCodes": [error["code"]],
        }
    document = source["document"]
    if not isinstance(document, dict):
        code = issues.add(source_path, "$", "reusable_block_not_object", "Reusable block record is not a JSON object.")
        return {
            "sourcePath": source_path,
            "sourceSha256": source["sourceSha256"],
            "status": "error",
            "classification": "unknown_reusable_block_record",
            "blockId": None,
            "name": None,
            "sourceRefs": None,
            "ports": {"inputs": [], "outputs": []},
            "graph": {"nodeIds": [], "edgeIds": [], "nodeCount": 0, "edgeCount": 0},
            "parameters": [],
            "prompts": [],
            "issueCodes": [code],
        }
    issue_codes = []
    if document.get("schemaVersion") == 2:
        classification = "block_definition_v2"
        block_id = _text(document.get("definitionId"))
        name = document.get("displayName")
        try:
            validate_block_definition_v2(document)
            status = "ok"
        except (TypeError, ValueError) as exc:
            status = "error"
            issue_codes.append(
                issues.add(
                    source_path,
                    "$",
                    "block_definition_v2_invalid",
                    f"BlockDefinitionV2 validation failed: {exc}",
                )
            )
        boundary = _object(document.get("boundary"))
        ports = {
            "inputs": copy.deepcopy(_array(boundary.get("inputs"))),
            "outputs": copy.deepcopy(_array(boundary.get("outputs"))),
        }
        graph_parameters = _graph_parameter_inventory(_object(document.get("graph")).get("nodes"))
        graph_value = _object(document.get("graph"))
        graph_nodes = _array(graph_value.get("nodes"))
        graph_edges = _array(graph_value.get("edges"))
        graph = {
            "nodeIds": [copy.deepcopy(_object(node).get("nodeId")) for node in graph_nodes],
            "edgeIds": [copy.deepcopy(_object(edge).get("edgeId")) for edge in graph_edges],
            "nodeCount": len(graph_nodes),
            "edgeCount": len(graph_edges),
        }
        controls = copy.deepcopy(_array(document.get("controls")))
        parameters = graph_parameters
        prompts = [
            {
                "controlId": control.get("controlId"),
                "defaultValuePresent": "defaultValue" in control,
                **({"defaultValue": copy.deepcopy(control["defaultValue"])} if "defaultValue" in control else {}),
            }
            for control in controls
            if isinstance(control, dict) and "prompt" in str(control.get("controlId") or "").lower()
        ]
        source_refs = {
            "source": copy.deepcopy(document.get("source")),
            "contentHash": copy.deepcopy(document.get("contentHash")),
            "graphHash": copy.deepcopy(_object(document.get("graph")).get("graphHash")),
            "ownership": copy.deepcopy(document.get("ownership")),
        }
        extra = {"controls": controls, "suggestedInputs": copy.deepcopy(document.get("suggestedInputs"))}
    elif document.get("version") == 1:
        classification = "legacy_user_block_definition_v1"
        block_id = _text(document.get("id"))
        name = document.get("name")
        status = "ok"
        for field in ("nodes", "edges", "inputs", "outputs", "exposedParams"):
            if not isinstance(document.get(field), list):
                status = "error"
                issue_codes.append(
                    issues.add(
                        source_path,
                        f"$.{field}",
                        "legacy_user_block_field_invalid",
                        f"Legacy User Node field {field} is not a list.",
                    )
                )
        ports = _declared_v1_ports(document)
        parameters = _graph_parameter_inventory(document.get("nodes"))
        graph_nodes = _array(document.get("nodes"))
        graph_edges = _array(document.get("edges"))
        graph = {
            "nodeIds": [copy.deepcopy(_object(node).get("id")) for node in graph_nodes],
            "edgeIds": [copy.deepcopy(_object(edge).get("id")) for edge in graph_edges],
            "nodeCount": len(graph_nodes),
            "edgeCount": len(graph_edges),
        }
        prompts = [prompt for node in parameters for prompt in node["prompts"]]
        source_refs = {"origin": copy.deepcopy(document.get("origin"))}
        extra = {"exposedParams": copy.deepcopy(document.get("exposedParams"))}
        for index, node_value in enumerate(_array(document.get("nodes"))):
            node = _object(node_value)
            if _root_families(node):
                status = "error"
                issue_codes.append(
                    issues.add(
                        source_path,
                        f"$.nodes[{index}]",
                        "nested_composite_in_reusable_definition",
                        "Reusable definition contains a nested composite marker.",
                    )
                )
    else:
        classification = "unknown_reusable_block_record"
        block_id = _text(document.get("id")) or _text(document.get("definitionId"))
        name = document.get("name") or document.get("displayName")
        status = "error"
        issue_codes.append(
            issues.add(
                source_path,
                "$",
                "reusable_block_schema_unknown",
                "Reusable block record is neither UserBlockDefinition V1 nor BlockDefinitionV2.",
            )
        )
        ports = {"inputs": [], "outputs": []}
        graph = {"nodeIds": [], "edgeIds": [], "nodeCount": 0, "edgeCount": 0}
        parameters = []
        prompts = []
        source_refs = None
        extra = {}
    if not block_id:
        status = "error"
        issue_codes.append(
            issues.add(source_path, "$", "reusable_block_id_missing", "Reusable block record has no stable ID.")
        )
    return {
        "sourcePath": source_path,
        "sourceSha256": source["sourceSha256"],
        "status": status,
        "classification": classification,
        "blockId": block_id,
        "name": copy.deepcopy(name),
        "sourceRefs": source_refs,
        "ports": ports,
        "graph": graph,
        "parameters": parameters,
        "prompts": prompts,
        "issueCodes": sorted(set(issue_codes)),
        **extra,
    }


def _definition_index(reusable_blocks: Sequence[dict[str, Any]], issues: _Issues) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for record in reusable_blocks:
        block_id = record.get("blockId")
        if isinstance(block_id, str) and block_id:
            result.setdefault(block_id, []).append(record)
    for block_id, records in sorted(result.items()):
        if len(records) > 1:
            for record in records:
                record["issueCodes"] = sorted(
                    {
                        *record["issueCodes"],
                        issues.add(
                            record["sourcePath"],
                            "$",
                            "reusable_block_id_duplicate",
                            f"Reusable block ID {block_id} appears in more than one source record.",
                        ),
                    }
                )
    return result


def _resolve_definition(
    composite: dict[str, Any],
    definition_index: Mapping[str, Sequence[dict[str, Any]]],
    source_path: str,
    issues: _Issues,
) -> None:
    classification = composite["classification"]
    if classification == "legacy_cluster_root":
        composite["definitionResolution"] = {
            "status": "external_registered_catalog_not_scanned",
            "matchedSourcePaths": [],
        }
        return
    if classification == "ambiguous_composite_root":
        composite["definitionResolution"] = {"status": "ambiguous_authority", "matchedSourcePaths": []}
        return
    if classification == "legacy_user_block_v1_root":
        block_id = _object(composite["sourceRefs"].get("legacyUserBlock")).get("userBlockId")
    else:
        block_id = _object(_object(composite["sourceRefs"].get("blockV2")).get("definitionRef")).get("definitionId")
        source_kind = _object(_object(composite["sourceRefs"].get("blockV2")).get("source")).get("kind")
        if source_kind in {"diffusers_catalog", "transformers_catalog"}:
            composite["definitionResolution"] = {
                "status": "external_registered_catalog_not_scanned",
                "matchedSourcePaths": [],
            }
            return
    matches = list(definition_index.get(block_id, [])) if isinstance(block_id, str) else []
    matched_paths = sorted(record["sourcePath"] for record in matches)
    if len(matches) == 1:
        status = "matched_reusable_store_record"
    elif len(matches) > 1:
        status = "ambiguous_reusable_store_records"
        composite["issueCodes"] = sorted(
            {
                *composite["issueCodes"],
                issues.add(
                    source_path,
                    f"composites.{composite.get('nodeId')}",
                    "composite_definition_resolution_ambiguous",
                    "Composite definition reference matches more than one reusable record.",
                ),
            }
        )
    else:
        status = "reusable_store_record_missing"
        composite["issueCodes"] = sorted(
            {
                *composite["issueCodes"],
                issues.add(
                    source_path,
                    f"composites.{composite.get('nodeId')}",
                    "composite_definition_record_missing",
                    "Composite definition reference is absent from the reusable User Node store.",
                    severity="warning",
                ),
            }
        )
    composite["definitionResolution"] = {"status": status, "matchedSourcePaths": matched_paths}


def _workflow_record(
    source: dict[str, Any],
    definition_index: Mapping[str, Sequence[dict[str, Any]]],
    issues: _Issues,
) -> dict[str, Any]:
    source_path = source["path"]
    base = {
        "sourcePath": source_path,
        "sourceSha256": source["sourceSha256"],
        "status": "ok",
        "readError": False,
        "workflowId": None,
        "title": None,
        "revision": None,
        "composites": [],
        "derivedChildren": [],
        "ordinaryNodeCount": 0,
        "issueCodes": [],
    }
    if source["readError"]:
        error = source["readError"]
        base["status"] = "error"
        base["readError"] = True
        base["issueCodes"] = [issues.add(source_path, "$", error["code"], error["message"])]
        return base
    document = source["document"]
    if not isinstance(document, dict):
        base["status"] = "error"
        base["issueCodes"] = [
            issues.add(source_path, "$", "workflow_record_not_object", "Saved workflow record is not a JSON object.")
        ]
        return base
    base.update(
        {
            "workflowId": copy.deepcopy(document.get("id")),
            "title": copy.deepcopy(document.get("title")),
            "revision": copy.deepcopy(document.get("revision")),
        }
    )
    snapshot = document.get("snapshot")
    if not isinstance(snapshot, dict):
        base["status"] = "error"
        base["issueCodes"] = [
            issues.add(
                source_path,
                "$.snapshot",
                "workflow_snapshot_missing",
                "Saved workflow record has no snapshot object.",
            )
        ]
        return base
    nodes_value = snapshot.get("nodes")
    edges_value = snapshot.get("edges")
    if not isinstance(nodes_value, list):
        base["status"] = "error"
        base["issueCodes"].append(
            issues.add(source_path, "$.snapshot.nodes", "workflow_nodes_invalid", "Workflow nodes are not a list.")
        )
    if not isinstance(edges_value, list):
        base["status"] = "error"
        base["issueCodes"].append(
            issues.add(source_path, "$.snapshot.edges", "workflow_edges_invalid", "Workflow edges are not a list.")
        )
    nodes = _array(nodes_value)
    edges = _array(edges_value)
    roots_by_id: dict[str, dict[str, Any]] = {}
    root_families_by_id: dict[str, list[str]] = {}
    node_ids: dict[str, int] = {}
    root_indexes = set()
    for index, node_value in enumerate(nodes):
        if not isinstance(node_value, dict):
            base["issueCodes"].append(
                issues.add(
                    source_path,
                    f"snapshot.nodes[{index}]",
                    "workflow_node_not_object",
                    "Workflow node is not a JSON object.",
                )
            )
            continue
        node_id = _text(node_value.get("id"))
        if node_id:
            if node_id in node_ids:
                base["issueCodes"].append(
                    issues.add(
                        source_path,
                        f"snapshot.nodes[{index}].id",
                        "workflow_node_id_duplicate",
                        f"Workflow node ID {node_id} is duplicated.",
                    )
                )
            else:
                node_ids[node_id] = index
        families = _root_families(node_value)
        if not families:
            continue
        root_indexes.add(index)
        root, root_families = _root_record(node_value, index, source_path, issues)
        base["composites"].append(root)
        if node_id:
            roots_by_id[node_id] = root
            root_families_by_id[node_id] = root_families
    for index, node_value in enumerate(nodes):
        if not isinstance(node_value, dict):
            continue
        owners = _derived_owners(node_value)
        if not owners:
            continue
        if index in root_indexes:
            base["issueCodes"].append(
                issues.add(
                    source_path,
                    f"snapshot.nodes[{index}]",
                    "composite_root_also_marked_as_derived",
                    "Composite root also contains a derived-child ownership marker.",
                )
            )
        if len(owners) > 1:
            base["issueCodes"].append(
                issues.add(
                    source_path,
                    f"snapshot.nodes[{index}]",
                    "derived_child_authority_ambiguous",
                    "Node contains more than one derived-child authority marker.",
                )
            )
        for kind, owner_id in owners:
            derived = _derived_record(node_value, index, kind, owner_id, source_path, issues)
            base["derivedChildren"].append(derived)
            root = roots_by_id.get(owner_id)
            if root is None:
                base["issueCodes"].append(
                    issues.add(
                        source_path,
                        f"snapshot.nodes[{index}]",
                        "derived_child_owner_missing",
                        f"Derived child owner {owner_id} is absent from the workflow.",
                    )
                )
                continue
            expected_family = {
                "legacy_cluster_child": "legacy_cluster",
                "legacy_user_block_child": "legacy_user_block_v1",
                "block_v2_projection": "block_v2",
            }[kind]
            if expected_family not in root_families_by_id.get(owner_id, []):
                base["issueCodes"].append(
                    issues.add(
                        source_path,
                        f"snapshot.nodes[{index}]",
                        "derived_child_owner_type_mismatch",
                        "Derived child owner exists but uses a different composite authority.",
                    )
                )
            child_id = _text(node_value.get("id"))
            if child_id:
                root["derivedChildIds"].append(child_id)
    base["ordinaryNodeCount"] = sum(
        isinstance(node, dict) and index not in root_indexes and not _derived_owners(node)
        for index, node in enumerate(nodes)
    )
    for root in base["composites"]:
        root_id = _text(root.get("nodeId"))
        owned_ids = {root_id} if root_id else set()
        owned_ids.update(root["derivedChildIds"])
        external_edges = []
        for edge_index, edge_value in enumerate(edges):
            if not isinstance(edge_value, dict):
                continue
            source_id = _text(edge_value.get("source"))
            target_id = _text(edge_value.get("target"))
            source_owned = source_id in owned_ids
            target_owned = target_id in owned_ids
            if source_owned == target_owned:
                continue
            external_edges.append(
                _edge_inventory(
                    edge_value,
                    "outbound" if source_owned else "inbound",
                    source_id if source_owned else target_id,
                )
            )
        root["derivedChildIds"] = sorted(set(root["derivedChildIds"]))
        root["externalEdges"] = sorted(
            external_edges,
            key=lambda edge: (
                str(edge.get("edgeId") or ""),
                edge["direction"],
                str(edge.get("sourceNodeId") or ""),
                str(edge.get("targetNodeId") or ""),
            ),
        )
        _resolve_definition(root, definition_index, source_path, issues)
    for edge_index, edge_value in enumerate(edges):
        if not isinstance(edge_value, dict):
            base["issueCodes"].append(
                issues.add(
                    source_path,
                    f"snapshot.edges[{edge_index}]",
                    "workflow_edge_not_object",
                    "Workflow edge is not a JSON object.",
                )
            )
            continue
        source_id = _text(edge_value.get("source"))
        target_id = _text(edge_value.get("target"))
        if not source_id or not target_id:
            base["issueCodes"].append(
                issues.add(
                    source_path,
                    f"snapshot.edges[{edge_index}]",
                    "workflow_edge_endpoint_invalid",
                    "Workflow edge is missing a source or target node ID.",
                )
            )
            continue
        missing = [node_id for node_id in (source_id, target_id) if node_id not in node_ids]
        if missing:
            base["issueCodes"].append(
                issues.add(
                    source_path,
                    f"snapshot.edges[{edge_index}]",
                    "workflow_edge_endpoint_missing",
                    f"Workflow edge references missing node(s): {', '.join(missing)}.",
                )
            )
    base["composites"] = sorted(base["composites"], key=lambda item: str(item.get("nodeId") or ""))
    base["derivedChildren"] = sorted(
        base["derivedChildren"],
        key=lambda item: (str(item.get("ownerId") or ""), str(item.get("nodeId") or ""), item["classification"]),
    )
    base["issueCodes"] = sorted(
        {
            *base["issueCodes"],
            *(item["code"] for item in issues.items if item["sourcePath"] == source_path),
        }
    )
    if base["issueCodes"]:
        base["status"] = "needs_review"
    return base


def _build_from_sources(
    workflow_sources: Sequence[dict[str, Any]],
    reusable_block_sources: Sequence[dict[str, Any]],
    *,
    initial_issues: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    issues = _Issues()
    issues.items.extend(copy.deepcopy(list(initial_issues)))
    reusable_blocks = [
        _reusable_record(source, issues) for source in sorted(reusable_block_sources, key=lambda x: x["path"])
    ]
    definition_index = _definition_index(reusable_blocks, issues)
    workflows = [
        _workflow_record(source, definition_index, issues)
        for source in sorted(workflow_sources, key=lambda x: x["path"])
    ]
    all_issues = issues.sorted()
    composite_records = [composite for workflow in workflows for composite in workflow["composites"]]
    derived_records = [child for workflow in workflows for child in workflow["derivedChildren"]]
    summary = {
        "workflowDocumentCount": len(workflows),
        "workflowReadErrorCount": sum(workflow["readError"] for workflow in workflows),
        "workflowErrorCount": sum(workflow["status"] == "error" for workflow in workflows),
        "reusableBlockDocumentCount": len(reusable_blocks),
        "reusableBlockReadErrorCount": sum(
            block["classification"] == "unreadable_reusable_block_record" for block in reusable_blocks
        ),
        "compositeRootCount": len(composite_records),
        "legacyClusterRootCount": sum(item["classification"] == "legacy_cluster_root" for item in composite_records),
        "legacyUserBlockV1RootCount": sum(
            item["classification"] == "legacy_user_block_v1_root" for item in composite_records
        ),
        "blockV2RootCount": sum(item["classification"] == "block_v2_root" for item in composite_records),
        "ambiguousCompositeRootCount": sum(
            item["classification"] == "ambiguous_composite_root" for item in composite_records
        ),
        "legacyClusterDerivedChildCount": sum(
            item["classification"] == "legacy_cluster_child" for item in derived_records
        ),
        "legacyUserBlockDerivedChildCount": sum(
            item["classification"] == "legacy_user_block_child" for item in derived_records
        ),
        "blockV2ProjectionChildCount": sum(
            item["classification"] == "block_v2_projection" for item in derived_records
        ),
        "orphanDerivedChildCount": sum(item["code"] == "derived_child_owner_missing" for item in all_issues),
        "legacyV1ReusableDefinitionCount": sum(
            item["classification"] == "legacy_user_block_definition_v1" for item in reusable_blocks
        ),
        "blockV2ReusableDefinitionCount": sum(
            item["classification"] == "block_definition_v2" for item in reusable_blocks
        ),
        "unknownReusableRecordCount": sum(
            item["classification"] == "unknown_reusable_block_record" for item in reusable_blocks
        ),
        "issueCount": len(all_issues),
        "errorCount": sum(item["severity"] == "error" for item in all_issues),
        "warningCount": sum(item["severity"] == "warning" for item in all_issues),
        "requiresManualReview": bool(all_issues),
    }
    report = {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "kind": "legacy_composite_migration_inventory",
        "mode": "read_only_dry_run",
        "boundary": {
            "readsSavedWorkflowJson": True,
            "readsReusableBlockJson": True,
            "writesFiles": False,
            "convertsRecords": False,
            "deletesRecords": False,
            "mergesRecords": False,
            "authorizesRecovery": False,
            "containsPromptAndParameterValues": True,
        },
        "sourceLayout": {
            "workflows": "user-workflows/*.json",
            "reusableBlocks": "studio/blocks/*.json",
        },
        "summary": summary,
        "workflows": workflows,
        "reusableBlocks": reusable_blocks,
        "issues": all_issues,
    }
    report["reportHash"] = _report_hash(report)
    return report


def build_composite_migration_inventory(
    workflow_documents: Mapping[str, Any],
    reusable_block_documents: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic report from caller-owned JSON values.

    Keys are stable source labels, normally paths relative to the data
    directory. Inputs are deep-copied and never mutated.
    """

    workflow_sources = [_source_record(path, document) for path, document in workflow_documents.items()]
    reusable_sources = [_source_record(path, document) for path, document in reusable_block_documents.items()]
    return _build_from_sources(workflow_sources, reusable_sources)


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant {value} is not supported.")


def _load_sources(data_dir: Path, relative_directory: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    directory = data_dir / relative_directory
    initial_issues = []
    if not directory.exists():
        return [], initial_issues
    if directory.is_symlink() or not directory.is_dir():
        initial_issues.append(
            {
                "sourcePath": relative_directory.as_posix(),
                "path": "$",
                "code": "source_directory_unsafe",
                "severity": "error",
                "message": "Inventory source directory is not a regular, non-symlink directory.",
            }
        )
        return [], initial_issues
    try:
        candidates = sorted(
            (path for path in directory.iterdir() if path.name.endswith(".json")),
            key=lambda path: path.name,
        )
    except OSError as exc:
        initial_issues.append(
            {
                "sourcePath": relative_directory.as_posix(),
                "path": "$",
                "code": "source_directory_unreadable",
                "severity": "error",
                "message": f"Inventory source directory could not be read ({type(exc).__name__}).",
            }
        )
        return [], initial_issues
    if len(candidates) > MAX_SOURCE_FILES:
        initial_issues.append(
            {
                "sourcePath": relative_directory.as_posix(),
                "path": "$",
                "code": "source_file_limit_exceeded",
                "severity": "error",
                "message": f"Inventory source contains more than {MAX_SOURCE_FILES} JSON files.",
            }
        )
        candidates = candidates[:MAX_SOURCE_FILES]
    sources = []
    total_source_bytes = 0
    for path in candidates:
        relative_path = (relative_directory / path.name).as_posix()
        if path.is_symlink() or not path.is_file():
            sources.append(
                _error_source_record(
                    relative_path,
                    source_sha256=None,
                    code="source_file_unsafe",
                    message="Inventory source is not a regular, non-symlink file.",
                )
            )
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            sources.append(
                _error_source_record(
                    relative_path,
                    source_sha256=None,
                    code="source_file_unreadable",
                    message=f"Inventory source metadata could not be read ({type(exc).__name__}).",
                )
            )
            continue
        if size > MAX_SOURCE_BYTES:
            sources.append(
                _error_source_record(
                    relative_path,
                    source_sha256=None,
                    code="source_file_too_large",
                    message=f"Inventory source exceeds the {MAX_SOURCE_BYTES}-byte limit.",
                )
            )
            continue
        if total_source_bytes + size > MAX_TOTAL_SOURCE_BYTES:
            sources.append(
                _error_source_record(
                    relative_path,
                    source_sha256=None,
                    code="source_total_size_limit_exceeded",
                    message=f"Inventory sources exceed the {MAX_TOTAL_SOURCE_BYTES}-byte aggregate limit.",
                )
            )
            continue
        try:
            raw = path.read_bytes()
        except OSError as exc:
            sources.append(
                _error_source_record(
                    relative_path,
                    source_sha256=None,
                    code="source_file_unreadable",
                    message=f"Inventory source could not be read ({type(exc).__name__}).",
                )
            )
            continue
        if len(raw) > MAX_SOURCE_BYTES:
            sources.append(
                _error_source_record(
                    relative_path,
                    source_sha256=None,
                    code="source_file_too_large",
                    message=f"Inventory source exceeds the {MAX_SOURCE_BYTES}-byte limit.",
                )
            )
            continue
        if total_source_bytes + len(raw) > MAX_TOTAL_SOURCE_BYTES:
            sources.append(
                _error_source_record(
                    relative_path,
                    source_sha256=None,
                    code="source_total_size_limit_exceeded",
                    message=f"Inventory sources exceed the {MAX_TOTAL_SOURCE_BYTES}-byte aggregate limit.",
                )
            )
            continue
        total_source_bytes += len(raw)
        source_sha256 = "sha256:" + hashlib.sha256(raw).hexdigest()
        try:
            document = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
        except UnicodeDecodeError:
            sources.append(
                _error_source_record(
                    relative_path,
                    source_sha256=source_sha256,
                    code="source_file_not_utf8",
                    message="Inventory source is not valid UTF-8.",
                )
            )
            continue
        except json.JSONDecodeError as exc:
            sources.append(
                _error_source_record(
                    relative_path,
                    source_sha256=source_sha256,
                    code="source_json_invalid",
                    message=f"Inventory source JSON is malformed at line {exc.lineno}, column {exc.colno}.",
                )
            )
            continue
        except (RecursionError, ValueError) as exc:
            sources.append(
                _error_source_record(
                    relative_path,
                    source_sha256=source_sha256,
                    code="source_json_invalid",
                    message=str(exc),
                )
            )
            continue
        sources.append(_source_record(relative_path, document, source_sha256))
    return sources, initial_issues


def scan_composite_migration_inventory(data_dir: str | Path) -> dict[str, Any]:
    """Read the two backend stores and return a report without writing them."""

    root = Path(data_dir)
    workflow_sources, workflow_issues = _load_sources(root, Path("user-workflows"))
    reusable_sources, reusable_issues = _load_sources(root, Path("studio/blocks"))
    return _build_from_sources(
        workflow_sources,
        reusable_sources,
        initial_issues=[*workflow_issues, *reusable_issues],
    )


def render_composite_migration_inventory(report: Mapping[str, Any], *, pretty: bool = True) -> str:
    """Render report JSON for stdout; no filesystem destination is accepted."""

    if pretty:
        return json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    return _canonical_json(report) + "\n"
