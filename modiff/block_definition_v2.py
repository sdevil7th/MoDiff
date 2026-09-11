"""Runtime contract for persisted composite block definitions.

``BlockDefinitionV2`` is the one reusable-definition format for User Nodes,
imported Hub blocks, and registered Diffusers/Transformers catalog entries.
``Cluster`` is catalog/provenance classification, not another definition or
canvas type.  Registered definitions use this exact in-memory contract but are
deliberately not writable through the user-owned ``/studio/blocks`` store.

The validator in this module is intentionally strict.  A persisted definition
is execution and connection metadata, so accepting a partially understood
shape would make an older backend reinterpret a newer contract.

Maintenance rule: reusable-definition fields, canonical JSON, and hashes must
change in lockstep with ``MoDiff-client/src/studio/blockSchemaV2.ts``, the
cross-runtime fixtures, API reference, and the normative unified-composite
contract. ``BlockInstanceV2`` remains workflow-owned, but backend migration
code emits its effective-interface hash through the shared helper in this
module so a newly persisted V2 instance never relies on legacy hydration.

Reviewed checkpoints that share one exact pipeline/workflow schema are stored
as ordinary bound instance controls. They do not mutate the registered
definition source or bypass its immutable artifact contract; execution must
resolve the selected value through a backend-reviewed repository/revision
allowlist. A checkpoint that changes graph structure requires another exact
registered definition.
"""

from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
import math
import re
from typing import Any, Literal, NotRequired, TypedDict


BlockSourceKindV2 = Literal[
    "diffusers_catalog",
    "transformers_catalog",
    "hub_import",
    "user",
]


class BlockSourceParentV2(TypedDict):
    definitionId: str
    contentHash: str
    sourceKind: BlockSourceKindV2


class BlockSourceV2(TypedDict):
    kind: BlockSourceKindV2
    catalogCategory: NotRequired[Literal["diffusers", "transformers"]]
    provider: NotRequired[str]
    library: NotRequired[Literal["diffusers", "transformers"]]
    libraryRevision: NotRequired[str]
    pipelineClass: NotRequired[str]
    blocksClass: NotRequired[str]
    workflow: NotRequired[str]
    manifestDefinitionId: NotRequired[str]
    manifestContentHash: NotRequired[str]
    executionAdmissionId: NotRequired[str]
    repository: NotRequired[str]
    repositoryRevision: NotRequired[str]
    parent: NotRequired[BlockSourceParentV2]


class BlockGraphNodeV2(TypedDict):
    nodeId: str
    nodeType: str
    data: dict[str, Any]
    semanticRole: NotRequired[str]
    upstreamBlockPath: NotRequired[str]
    modularDiffusers: NotRequired["BlockGraphNodeModularDiffusersV2"]
    containerInterface: NotRequired["BlockContainerInterfaceV1"]
    parentNodeId: NotRequired[str]


class BlockGraphNodeModularDiffusersV2(TypedDict):
    """Pinned identity for infrastructure or one exact upstream block.

    This metadata is part of the canonical graph hash. It describes the
    semantic node shown in the editor; values remain exclusively in
    ``BlockGraphNodeV2.data.params`` / ``BlockInstanceV2.values``.
    """

    kind: Literal["infrastructure", "upstream_block"]
    pipelineClass: str
    blocksClass: str
    workflowId: str
    libraryRevision: str
    runtimeRole: str
    blockDefinitionId: NotRequired[str]
    blockClass: NotRequired[str]
    blockKind: NotRequired[Literal["auto", "conditional", "sequential", "loop", "block"]]
    blockContractHash: NotRequired[str]
    placementPath: NotRequired[list[str]]
    parentPlacementPath: NotRequired[list[str]]
    sourceDefinitionId: NotRequired[str]
    sourcePlacementPath: NotRequired[list[str]]
    sourceExecutionScope: NotRequired[Literal["selected_workflow", "unpruned_pipeline"]]
    componentNames: NotRequired[list[str]]


class BlockGraphEdgeV2(TypedDict):
    edgeId: str
    sourceNodeId: str
    sourcePortId: str
    targetNodeId: str
    targetPortId: str


class BlockGraphV2(TypedDict):
    nodes: list[BlockGraphNodeV2]
    edges: list[BlockGraphEdgeV2]
    executionOrder: NotRequired[list[str]]
    graphHash: str


class BlockPortBindingV2(TypedDict):
    nodeId: str
    fieldOrPortId: str


class BlockPortV2(TypedDict):
    portId: str
    label: str
    valueType: str
    required: bool
    multiple: NotRequired[bool]
    binding: BlockPortBindingV2
    mirrorBindings: NotRequired[list[BlockPortBindingV2]]


class BlockBoundaryDerivationV2(TypedDict):
    algorithmVersion: str
    derivedAtDefinitionHash: str


class BlockBoundaryV2(TypedDict):
    mode: Literal["explicit", "derived"]
    inputs: list[BlockPortV2]
    outputs: list[BlockPortV2]
    derivation: NotRequired[BlockBoundaryDerivationV2]


class BlockControlBindingV2(TypedDict):
    nodeId: str
    fieldId: str


class BlockControlV2(TypedDict):
    controlId: str
    label: str
    binding: BlockControlBindingV2
    mirrorBindings: NotRequired[list[BlockControlBindingV2]]
    valueType: str
    defaultValue: NotRequired[Any]
    required: NotRequired[bool]
    sealed: NotRequired[bool]
    order: int
    group: NotRequired[str]
    help: NotRequired[str]


class BlockContainerControlV1(TypedDict):
    """A field view with no independent value/default authority."""

    controlId: str
    label: str
    binding: BlockControlBindingV2
    mirrorBindings: NotRequired[list[BlockControlBindingV2]]
    valueType: str
    required: NotRequired[bool]
    sealed: NotRequired[bool]
    order: int
    group: NotRequired[str]
    help: NotRequired[str]


class BlockContainerInterfaceV1(TypedDict):
    schemaVersion: Literal[1]
    boundary: BlockBoundaryV2
    controls: list[BlockContainerControlV1]
    previews: NotRequired[list["BlockPreviewBindingV2"]]


class SuggestedInputSetV2(TypedDict):
    suggestionId: str
    label: str
    source: NotRequired[str]
    values: dict[str, Any]


class BlockPreviewBindingV2(TypedDict):
    nodeId: str
    outputPortId: str
    mediaType: Literal["image", "video", "audio", "text", "file"]
    primary: NotRequired[bool]


class BlockOwnershipV2(TypedDict):
    kind: Literal["registered", "user"]
    definitionMutable: bool


class BlockDefinitionV2(TypedDict):
    schemaVersion: Literal[2]
    definitionId: str
    displayName: str
    description: NotRequired[str]
    contentHash: str
    source: BlockSourceV2
    graph: BlockGraphV2
    boundary: BlockBoundaryV2
    controls: list[BlockControlV2]
    suggestedInputs: NotRequired[list[SuggestedInputSetV2]]
    previews: list[BlockPreviewBindingV2]
    ownership: BlockOwnershipV2


class BlockEffectiveInterfaceV2(TypedDict):
    """Workflow-instance surface whose hash contract is shared with the client."""

    boundary: BlockBoundaryV2
    controls: list[BlockControlV2]
    baseInterfaceHash: str
    effectiveInterfaceHash: str


class BlockRouteDraftV1(TypedDict):
    """Inactive exact-route state; deliberately not a nested block instance."""

    schemaVersion: Literal[1]
    routeKey: str
    definitionRef: dict[str, str]
    definitionSnapshot: BlockDefinitionV2
    effectiveGraph: BlockGraphV2
    effectiveInterface: BlockEffectiveInterfaceV2
    values: dict[str, Any]
    customization: dict[str, Any]
    internalLayout: dict[str, Any]
    internalLayoutMode: NotRequired[Literal["root", "hierarchical"]]
    collapsedContainerNodeIds: NotRequired[list[str]]


class BlockRouteSelectionV1(TypedDict):
    schemaVersion: Literal[1]
    routeSetId: str
    selectedRouteKey: str
    inactiveDrafts: dict[str, BlockRouteDraftV1]


class BlockInstanceV2(TypedDict):
    """Workflow-owned V2 instance accepted by backend execution authorities."""

    schemaVersion: Literal[2]
    instanceId: str
    definitionRef: dict[str, str]
    definitionSnapshot: BlockDefinitionV2
    effectiveGraph: BlockGraphV2
    effectiveInterface: BlockEffectiveInterfaceV2
    values: dict[str, Any]
    customization: dict[str, Any]
    presentation: dict[str, Any]
    previewStates: list[dict[str, Any]]
    authorities: list[dict[str, Any]]
    routeSelection: NotRequired[BlockRouteSelectionV1]


def modular_container_node_ids_v2(graph: BlockGraphV2 | dict[str, Any]) -> list[str]:
    """Return upstream placements that own immediate Modular children.

    Container identity comes from the pinned placement hierarchy, not from a
    canvas node type. An executable ``custom`` upstream block may itself own
    sequential or loop members and must therefore be independently
    collapsible in the editor.
    """

    return sorted(set(block_graph_parent_ids_v2(graph).values()) | {
        node["nodeId"] for node in graph["nodes"] if node["nodeType"] == "group" and not node.get("modularDiffusers")
    })


def block_graph_parent_ids_v2(graph: BlockGraphV2 | dict[str, Any]) -> dict[str, str]:
    """Source-neutral customized ownership, falling back to exact catalog placement."""
    node_id_by_placement: dict[str, str] = {}
    nodes_by_id = {node["nodeId"]: node for node in graph.get("nodes", [])}
    for node in graph.get("nodes", []):
        metadata = node.get("modularDiffusers")
        if not isinstance(metadata, dict) or metadata.get("kind") != "upstream_block":
            continue
        placement = metadata.get("placementPath")
        if isinstance(placement, list) and placement:
            key = "/".join(placement)
            if key in node_id_by_placement:
                raise ValueError(f"Ambiguous Block V2 subtree placement {key}.")
            node_id_by_placement[key] = node["nodeId"]

    parents: dict[str, str] = {}
    for node in graph.get("nodes", []):
        metadata = node.get("modularDiffusers", {})
        parent_path = metadata.get("parentPlacementPath")
        upstream_parent = (
            node_id_by_placement.get("/".join(parent_path))
            if metadata.get("kind") == "upstream_block" and parent_path else None
        )
        parent_id = node.get("parentNodeId", upstream_parent)
        if not parent_id:
            continue
        if "parentNodeId" in node:
            parent = nodes_by_id.get(parent_id)
            if parent is None:
                raise ValueError(f"Block V2 node {node['nodeId']} references unknown parent {parent_id}.")
            parent_metadata = parent.get("modularDiffusers", {})
            if parent["nodeType"] != "group" and not (
                parent_metadata.get("kind") == "upstream_block" and parent_metadata.get("blockKind") != "block"
            ):
                raise ValueError(f"Block V2 parent {parent_id} is not a container.")
            if upstream_parent and upstream_parent != parent_id:
                raise ValueError(f"Block V2 node {node['nodeId']} has conflicting explicit and upstream parents.")
        parents[node["nodeId"]] = parent_id
    for node_id in parents:
        seen: set[str] = set()
        current = node_id
        while current in parents:
            if current in seen:
                raise ValueError(f"Cyclic Block V2 subtree at {current}.")
            seen.add(current)
            current = parents[current]
    return parents


_SOURCE_KINDS = {"diffusers_catalog", "transformers_catalog", "hub_import", "user"}
_CATALOG_SOURCE_KINDS = {"diffusers_catalog", "transformers_catalog"}
_USER_STORE_SOURCE_KINDS = {"hub_import", "user"}
_MEDIA_TYPES = {"image", "video", "audio", "text", "file"}
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,383}$")
# Public ids remain stricter, while bindings preserve exact backend/Python
# parameter keys such as Modular Diffusers `_auto_resize`.
_FIELD_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:/-]{0,383}$")
_REPOSITORY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_HASH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{1,511}$")
_IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_MAX_DEFINITION_BYTES = 16 * 1024 * 1024
_MAX_INSTANCE_BYTES = 20 * 1024 * 1024
_COMPOSITE_NODE_TYPES = {"block", "cluster"}
_COMPOSITE_DATA_MARKERS = {
    "blockDefinitionId",
    "blockDefinitionV2",
    "blockInstanceId",
    "blockInstanceV2",
    "definitionSnapshot",
    "huggingFaceClusterDefinitionId",
    "huggingFaceClusterInstanceId",
    "huggingFaceClusterSnapshot",
    "userBlockId",
    "userBlockInstanceId",
    "userBlockSnapshot",
}


def _utf16_sort_key(value: str) -> bytes:
    """Sort strings by the UTF-16 code units used by JavaScript ``<``."""

    return value.encode("utf-16-be", errors="surrogatepass")


def _javascript_number(value: int | float) -> str:
    """Return the JSON number spelling used by ECMAScript for finite values."""

    number = float(value)
    if not math.isfinite(number):
        raise ValueError("BlockDefinitionV2 canonical JSON requires finite numbers.")
    if number == 0:
        return "0"
    sign = "-" if number < 0 else ""
    raw = repr(abs(number)).lower()
    coefficient, separator, exponent_text = raw.partition("e")
    exponent = int(exponent_text) if separator else 0
    integer_part, dot, fraction_part = coefficient.partition(".")
    digits = (integer_part + (fraction_part if dot else "")).lstrip("0").rstrip("0")
    if not digits:
        return "0"
    decimal_position = len(integer_part.lstrip("0")) + exponent
    if integer_part == "0":
        leading_fraction_zeros = len(fraction_part) - len(fraction_part.lstrip("0"))
        decimal_position = -leading_fraction_zeros + exponent

    if 0 < decimal_position <= 21:
        if decimal_position >= len(digits):
            body = digits + "0" * (decimal_position - len(digits))
        else:
            body = digits[:decimal_position] + "." + digits[decimal_position:]
    elif -6 < decimal_position <= 0:
        body = "0." + "0" * (-decimal_position) + digits
    else:
        body = digits[0]
        if len(digits) > 1:
            body += "." + digits[1:]
        scientific_exponent = decimal_position - 1
        body += f"e{'+' if scientific_exponent >= 0 else ''}{scientific_exponent}"
    return sign + body


def _javascript_string(value: str) -> str:
    """Return the well-formed JSON string spelling used by ``JSON.stringify``."""

    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return "".join(
        f"\\u{ord(character):04x}" if 0xD800 <= ord(character) <= 0xDFFF else character for character in encoded
    )


def _stable_json(value: Any) -> str:
    """Mirror V2 client canonical JSON, including UTF-16 key ordering."""

    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        if abs(value) <= 2**53 - 1:
            return str(value)
        canonical = _javascript_number(value)
        if canonical != str(value):
            raise ValueError(
                "BlockDefinitionV2 integers outside JavaScript's exact range must use their "
                "canonical JSON number spelling."
            )
        return canonical
    if isinstance(value, float):
        return _javascript_number(value)
    if isinstance(value, str):
        return _javascript_string(value)
    if isinstance(value, list):
        return "[" + ",".join(_stable_json(item) for item in value) + "]"
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("BlockDefinitionV2 canonical JSON object keys must be strings.")
        entries = []
        for key in sorted(value, key=_utf16_sort_key):
            entries.append(f"{_stable_json(key)}:{_stable_json(value[key])}")
        return "{" + ",".join(entries) + "}"
    raise ValueError("BlockDefinitionV2 canonical JSON accepts only JSON values.")


def _hash_string(value: str) -> str:
    """Mirror the client's FNV-1a loop over JavaScript UTF-16 code units."""

    digest = 0x811C9DC5
    encoded = value.encode("utf-16-le", errors="surrogatepass")
    for index in range(0, len(encoded), 2):
        digest ^= encoded[index] | encoded[index + 1] << 8
        digest = digest * 0x01000193 & 0xFFFFFFFF
    return f"{digest:08x}"


def canonical_block_graph_v2(graph: dict[str, Any]) -> dict[str, Any]:
    """Return the execution-semantic graph payload shared with the client."""

    return {
        "nodes": sorted(copy.deepcopy(graph["nodes"]), key=lambda node: _utf16_sort_key(node["nodeId"])),
        "edges": sorted(copy.deepcopy(graph["edges"]), key=lambda edge: _utf16_sort_key(edge["edgeId"])),
        **({"executionOrder": copy.deepcopy(graph["executionOrder"])} if "executionOrder" in graph else {}),
    }


def block_graph_hash_v2(graph: dict[str, Any]) -> str:
    return f"block-graph-v2-{_hash_string(_stable_json(canonical_block_graph_v2(graph)))}"


def _execution_relevant_source_v2(source: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "provider",
        "library",
        "libraryRevision",
        "pipelineClass",
        "blocksClass",
        "workflow",
        "manifestDefinitionId",
        "manifestContentHash",
        "executionAdmissionId",
        "repository",
        "repositoryRevision",
    )
    return {key: copy.deepcopy(source[key]) for key in keys if source.get(key)}


def canonical_block_definition_v2(definition: dict[str, Any]) -> dict[str, Any]:
    """Return fields covered by ``BlockDefinitionV2.contentHash``."""

    return {
        "graph": canonical_block_graph_v2(definition["graph"]),
        "boundary": copy.deepcopy(definition["boundary"]),
        "controls": copy.deepcopy(definition["controls"]),
        "previews": copy.deepcopy(definition["previews"]),
        "sourceBindings": _execution_relevant_source_v2(definition["source"]),
    }


def block_definition_content_hash_v2(definition: dict[str, Any]) -> str:
    return "block-definition-v2-" + _hash_string(_stable_json(canonical_block_definition_v2(definition)))


def block_definition_canonical_sha256_v2(definition: dict[str, Any]) -> str:
    """Collision-resistant digest for backend execution-authority pinning."""

    canonical = _stable_json(canonical_block_definition_v2(definition)).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def canonical_block_interface_v2(value: dict[str, Any]) -> dict[str, Any]:
    """Return the instance surface covered by the V2 interface hash."""

    return {
        "boundary": copy.deepcopy(value["boundary"]),
        "controls": copy.deepcopy(value["controls"]),
    }


def block_interface_hash_v2(value: dict[str, Any]) -> str:
    """Mirror the client's ``blockInterfaceHashV2`` exactly."""

    return "block-interface-v2-" + _hash_string(_stable_json(canonical_block_interface_v2(value)))


def block_execution_parameter_hash_v2(instance: dict[str, Any]) -> str:
    """Return the values/interface hash bound into a V2 execution receipt.

    Callers must validate the instance first. Keeping this canonical payload
    intentionally small mirrors ``blockExecutionParameterHashV2`` in the
    client and prevents presentation, preview progress, or old receipts from
    invalidating an otherwise identical execution.
    """

    return "block-execution-parameters-v2-" + _hash_string(
        _stable_json(
            {
                "values": copy.deepcopy(instance["values"]),
                "effectiveInterfaceHash": instance["effectiveInterface"]["effectiveInterfaceHash"],
            }
        )
    )


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a JSON object.")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list.")
    return value


def _exact_keys(
    value: dict[str, Any],
    path: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} keys must be strings.")
    value_keys = set(value)
    missing = required - value_keys
    unknown = value_keys - required - optional
    if missing:
        raise ValueError(f"{path} is missing required field(s): {', '.join(sorted(missing))}.")
    if unknown:
        raise ValueError(f"{path} has unsupported field(s): {', '.join(sorted(unknown))}.")


def _string(
    value: Any,
    path: str,
    *,
    maximum: int = 512,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-16-le", errors="surrogatepass")) // 2 > maximum
        or (pattern is not None and pattern.fullmatch(value) is None)
    ):
        raise ValueError(f"{path} is malformed.")
    return value


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean.")
    return value


def _integer(value: Any, path: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be an integer.")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise ValueError(f"{path} must be an integer.")
    return value


def _finite_number(value: Any, path: str, *, minimum: float | None = None) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{path} must be a finite number.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be at least {minimum}.")
    return value


def _json_value(value: Any, path: str, seen: set[int] | None = None) -> None:
    seen = seen or set()
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, str):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity.")
        return
    if isinstance(value, list):
        identity = id(value)
        if identity in seen:
            raise ValueError(f"{path} must contain only acyclic JSON values.")
        seen.add(identity)
        for index, item in enumerate(value):
            _json_value(item, f"{path}[{index}]", seen)
        seen.remove(identity)
        return
    if isinstance(value, dict):
        identity = id(value)
        if identity in seen:
            raise ValueError(f"{path} must contain only acyclic JSON values.")
        seen.add(identity)
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings.")
            _json_value(item, f"{path}.{key}", seen)
        seen.remove(identity)
        return
    raise ValueError(f"{path} must contain only JSON values.")


def _validate_source(source_value: Any) -> dict[str, Any]:
    source = _object(source_value, "BlockDefinitionV2.source")
    _exact_keys(
        source,
        "BlockDefinitionV2.source",
        required={"kind"},
        optional={
            "catalogCategory",
            "provider",
            "library",
            "libraryRevision",
            "pipelineClass",
            "blocksClass",
            "workflow",
            "manifestDefinitionId",
            "manifestContentHash",
            "executionAdmissionId",
            "repository",
            "repositoryRevision",
            "parent",
        },
    )
    kind = _string(source["kind"], "BlockDefinitionV2.source.kind")
    if kind not in _SOURCE_KINDS:
        raise ValueError("BlockDefinitionV2.source.kind is unsupported.")

    for key in (
        "provider",
        "libraryRevision",
        "pipelineClass",
        "blocksClass",
        "workflow",
        "manifestDefinitionId",
        "manifestContentHash",
        "executionAdmissionId",
        "repository",
        "repositoryRevision",
    ):
        if key in source:
            _string(source[key], f"BlockDefinitionV2.source.{key}")

    if "catalogCategory" in source:
        category = _string(source["catalogCategory"], "BlockDefinitionV2.source.catalogCategory")
        if category not in {"diffusers", "transformers"}:
            raise ValueError("BlockDefinitionV2.source.catalogCategory is unsupported.")
    if "library" in source:
        library = _string(source["library"], "BlockDefinitionV2.source.library")
        if library not in {"diffusers", "transformers"}:
            raise ValueError("BlockDefinitionV2.source.library is unsupported.")
    if kind == "diffusers_catalog" and (
        source.get("catalogCategory") != "diffusers" or source.get("library") != "diffusers"
    ):
        raise ValueError("A Diffusers catalog BlockDefinitionV2 source requires Diffusers category and library.")
    if kind == "transformers_catalog" and (
        source.get("catalogCategory") != "transformers" or source.get("library") != "transformers"
    ):
        raise ValueError("A Transformers catalog BlockDefinitionV2 source requires Transformers category and library.")
    if kind in _USER_STORE_SOURCE_KINDS and "catalogCategory" in source:
        raise ValueError("User and Hub-import BlockDefinitionV2 sources cannot use a catalog category.")

    repository = source.get("repository")
    revision = source.get("repositoryRevision")
    if (repository is None) != (revision is None):
        raise ValueError("BlockDefinitionV2.source.repository and repositoryRevision must be provided together.")
    if repository is not None:
        _string(
            repository,
            "BlockDefinitionV2.source.repository",
            maximum=384,
            pattern=_REPOSITORY_ID_RE,
        )
        _string(
            revision,
            "BlockDefinitionV2.source.repositoryRevision",
            maximum=40,
            pattern=_IMMUTABLE_REVISION_RE,
        )
    if kind == "hub_import" and repository is None:
        raise ValueError("BlockDefinitionV2.source kind hub_import requires repository and repositoryRevision.")
    if kind in _CATALOG_SOURCE_KINDS and any(
        not source.get(key)
        for key in (
            "libraryRevision",
            "manifestDefinitionId",
            "manifestContentHash",
            "pipelineClass",
            "workflow",
        )
    ):
        raise ValueError(
            "Registered catalog BlockDefinitionV2 sources require library, manifest, pipeline class, "
            "and workflow identities."
        )

    if "parent" in source:
        parent = _object(source["parent"], "BlockDefinitionV2.source.parent")
        _exact_keys(
            parent,
            "BlockDefinitionV2.source.parent",
            required={"definitionId", "contentHash", "sourceKind"},
        )
        _string(
            parent["definitionId"],
            "BlockDefinitionV2.source.parent.definitionId",
            maximum=384,
            pattern=_ID_RE,
        )
        _string(
            parent["contentHash"],
            "BlockDefinitionV2.source.parent.contentHash",
            pattern=_HASH_RE,
        )
        parent_source_kind = _string(parent["sourceKind"], "BlockDefinitionV2.source.parent.sourceKind")
        if parent_source_kind not in _SOURCE_KINDS:
            raise ValueError("BlockDefinitionV2.source.parent.sourceKind is unsupported.")

    return source


def _contains_composite_node(node_type: str, data: dict[str, Any]) -> bool:
    if node_type.casefold() in _COMPOSITE_NODE_TYPES:
        return True
    data_type = data.get("type")
    if isinstance(data_type, str) and data_type.casefold() in _COMPOSITE_NODE_TYPES:
        return True
    return any(marker in data for marker in _COMPOSITE_DATA_MARKERS)


def _validate_modular_diffusers_node(value: Any, path: str) -> dict[str, Any]:
    metadata = _object(value, path)
    common = {
        "kind",
        "pipelineClass",
        "blocksClass",
        "workflowId",
        "libraryRevision",
        "runtimeRole",
    }
    upstream = {
        "blockDefinitionId",
        "blockClass",
        "blockKind",
        "blockContractHash",
        "placementPath",
        "parentPlacementPath",
        "sourceDefinitionId",
        "sourcePlacementPath",
        "sourceExecutionScope",
        "componentNames",
    }
    kind = metadata.get("kind")
    if kind == "infrastructure":
        _exact_keys(metadata, path, required=common, optional={"componentNames"})
    elif kind == "upstream_block":
        _exact_keys(
            metadata,
            path,
            required=common
            | (
                upstream
                - {
                    "parentPlacementPath",
                    "sourceDefinitionId",
                    "sourcePlacementPath",
                    "sourceExecutionScope",
                }
            ),
            optional={
                "parentPlacementPath",
                "sourceDefinitionId",
                "sourcePlacementPath",
                "sourceExecutionScope",
            },
        )
    else:
        raise ValueError(f"{path}.kind is unsupported.")

    for key in ("pipelineClass", "blocksClass", "runtimeRole"):
        _string(metadata[key], f"{path}.{key}", maximum=512, pattern=_ID_RE)
    # Exact Modular Diffusers workflow ids, placement paths, and component
    # names are Python identifiers and may intentionally begin with `_`.
    _string(metadata["workflowId"], f"{path}.workflowId", maximum=512, pattern=_FIELD_ID_RE)
    _string(metadata["libraryRevision"], f"{path}.libraryRevision", pattern=_IMMUTABLE_REVISION_RE)

    def string_array(key: str, *, required_nonempty: bool = False) -> list[str]:
        values = _array(metadata.get(key, []), f"{path}.{key}")
        if required_nonempty and not values:
            raise ValueError(f"{path}.{key} must not be empty.")
        if len(values) > 256:
            raise ValueError(f"{path}.{key} exceeds the item limit.")
        normalized = [
            _string(item, f"{path}.{key}[{index}]", maximum=512, pattern=_FIELD_ID_RE)
            for index, item in enumerate(values)
        ]
        if len(normalized) != len(set(normalized)) and key == "componentNames":
            raise ValueError(f"{path}.{key} must not contain duplicates.")
        return normalized

    if "componentNames" in metadata:
        string_array("componentNames")
    if kind == "upstream_block":
        _string(metadata["blockDefinitionId"], f"{path}.blockDefinitionId", maximum=512, pattern=_ID_RE)
        _string(metadata["blockClass"], f"{path}.blockClass", maximum=512, pattern=_ID_RE)
        if metadata["blockKind"] not in {"auto", "conditional", "sequential", "loop", "block"}:
            raise ValueError(f"{path}.blockKind is unsupported.")
        _string(metadata["blockContractHash"], f"{path}.blockContractHash", pattern=_HASH_RE)
        string_array("placementPath", required_nonempty=True)
        if "parentPlacementPath" in metadata:
            string_array("parentPlacementPath")
        if "sourceDefinitionId" in metadata:
            _string(metadata["sourceDefinitionId"], f"{path}.sourceDefinitionId", maximum=512, pattern=_ID_RE)
        if "sourcePlacementPath" in metadata:
            string_array("sourcePlacementPath", required_nonempty=True)
        if "sourceExecutionScope" in metadata and metadata["sourceExecutionScope"] not in {
            "selected_workflow",
            "unpruned_pipeline",
        }:
            raise ValueError(f"{path}.sourceExecutionScope is unsupported.")
        source_fields = sum(
            key in metadata
            for key in ("sourceDefinitionId", "sourcePlacementPath", "sourceExecutionScope")
        )
        if source_fields not in {0, 3}:
            raise ValueError(
                f"{path} source definition, placement, and execution scope must be declared together."
            )
    return metadata


def _validate_graph(graph_value: Any) -> tuple[dict[str, Any], set[str]]:
    graph = _object(graph_value, "BlockDefinitionV2.graph")
    _exact_keys(
        graph,
        "BlockDefinitionV2.graph",
        required={"nodes", "edges", "graphHash"},
        optional={"executionOrder"},
    )
    _string(graph["graphHash"], "BlockDefinitionV2.graph.graphHash", pattern=_HASH_RE)

    nodes = _array(graph["nodes"], "BlockDefinitionV2.graph.nodes")
    node_ids: set[str] = set()
    for index, node_value in enumerate(nodes):
        path = f"BlockDefinitionV2.graph.nodes[{index}]"
        node = _object(node_value, path)
        _exact_keys(
            node,
            path,
            required={"nodeId", "nodeType", "data"},
            optional={"semanticRole", "upstreamBlockPath", "modularDiffusers", "containerInterface", "parentNodeId"},
        )
        node_id = _string(node["nodeId"], f"{path}.nodeId", maximum=384, pattern=_ID_RE)
        node_type = _string(node["nodeType"], f"{path}.nodeType")
        if "parentNodeId" in node:
            _string(node["parentNodeId"], f"{path}.parentNodeId", maximum=384, pattern=_ID_RE)
        if node_id in node_ids:
            raise ValueError(f"BlockDefinitionV2.graph contains duplicate nodeId {node_id!r}.")
        node_ids.add(node_id)
        data = _object(node["data"], f"{path}.data")
        _json_value(data, f"{path}.data")
        if _contains_composite_node(node_type, data):
            raise ValueError("Nested composite nodes are not supported. Flatten the block before saving.")
        for key in ("semanticRole", "upstreamBlockPath"):
            if key in node:
                _string(
                    node[key],
                    f"{path}.{key}",
                    maximum=2048 if key == "upstreamBlockPath" else 512,
                )
        if "modularDiffusers" in node:
            _validate_modular_diffusers_node(node["modularDiffusers"], f"{path}.modularDiffusers")

    edges = _array(graph["edges"], "BlockDefinitionV2.graph.edges")
    edge_ids: set[str] = set()
    for index, edge_value in enumerate(edges):
        path = f"BlockDefinitionV2.graph.edges[{index}]"
        edge = _object(edge_value, path)
        _exact_keys(
            edge,
            path,
            required={"edgeId", "sourceNodeId", "sourcePortId", "targetNodeId", "targetPortId"},
        )
        edge_id = _string(edge["edgeId"], f"{path}.edgeId", maximum=384, pattern=_ID_RE)
        if edge_id in edge_ids:
            raise ValueError(f"BlockDefinitionV2.graph contains duplicate edgeId {edge_id!r}.")
        edge_ids.add(edge_id)
        for key in ("sourceNodeId", "sourcePortId", "targetNodeId", "targetPortId"):
            _string(edge[key], f"{path}.{key}", maximum=384, pattern=_ID_RE)
        if edge["sourceNodeId"] not in node_ids or edge["targetNodeId"] not in node_ids:
            raise ValueError(f"{path} must reference nodes declared in BlockDefinitionV2.graph.nodes.")

    if "executionOrder" in graph:
        order = _array(graph["executionOrder"], "BlockDefinitionV2.graph.executionOrder")
        for index, node_id in enumerate(order):
            _string(
                node_id,
                f"BlockDefinitionV2.graph.executionOrder[{index}]",
                maximum=384,
                pattern=_ID_RE,
            )
        if len(order) != len(set(order)):
            raise ValueError("BlockDefinitionV2.graph.executionOrder must not contain duplicates.")
        if not set(order).issubset(node_ids):
            raise ValueError("BlockDefinitionV2.graph.executionOrder references an unknown graph node.")

    block_graph_parent_ids_v2(graph)
    for node in nodes:
        if "containerInterface" in node:
            local = validate_block_container_interface_v1(node["containerInterface"], graph, node["nodeId"])
            included = block_graph_subtree_node_ids_v2(graph, node["nodeId"])
            for edge in edges:
                for direction in ("input", "output"):
                    endpoint_id = edge["targetNodeId" if direction == "input" else "sourceNodeId"]
                    opposite_id = edge["sourceNodeId" if direction == "input" else "targetNodeId"]
                    field_id = edge["targetPortId" if direction == "input" else "sourcePortId"]
                    if endpoint_id not in included or (endpoint_id != node["nodeId"] and opposite_id in included):
                        continue
                    ports = local["boundary"]["inputs" if direction == "input" else "outputs"]
                    if not any(
                        binding["nodeId"] == endpoint_id and binding["fieldOrPortId"] == field_id
                        for port in ports for binding in [port["binding"], *port.get("mirrorBindings", [])]
                    ):
                        # A live crossing is displayed as a connected-only socket;
                        # it does not become part of the reusable interface.
                        endpoint = next(item for item in nodes if item["nodeId"] == endpoint_id)
                        params = endpoint.get("data", {}).get("params", {})
                        field = params.get(field_id) if isinstance(params, dict) else None
                        if not isinstance(field, dict) or (field.get("display") == "output") != (direction == "output"):
                            raise ValueError(
                                f"BlockDefinitionV2.graph container {node['nodeId']} has an invalid connected {direction} {endpoint_id}.{field_id}."
                            )

    expected_hash = block_graph_hash_v2(graph)
    if graph["graphHash"] != expected_hash:
        raise ValueError(f"BlockDefinitionV2.graph.graphHash must be {expected_hash}.")

    return graph, node_ids


def _validate_port(
    port_value: Any,
    path: str,
    *,
    graph: dict[str, Any],
    direction: Literal["input", "output"],
) -> str:
    nodes_by_id = {node["nodeId"]: node for node in graph["nodes"]}
    node_ids = set(nodes_by_id)
    port = _object(port_value, path)
    _exact_keys(
        port,
        path,
        required={"portId", "label", "valueType", "required", "binding"},
        optional={"multiple", "mirrorBindings"},
    )
    port_id = _string(port["portId"], f"{path}.portId", maximum=384, pattern=_ID_RE)
    _string(port["label"], f"{path}.label")
    _string(port["valueType"], f"{path}.valueType")
    _bool(port["required"], f"{path}.required")
    if "multiple" in port:
        _bool(port["multiple"], f"{path}.multiple")
    binding = _object(port["binding"], f"{path}.binding")
    _exact_keys(binding, f"{path}.binding", required={"nodeId", "fieldOrPortId"})
    _string(binding["nodeId"], f"{path}.binding.nodeId", maximum=384, pattern=_ID_RE)
    _string(
        binding["fieldOrPortId"],
        f"{path}.binding.fieldOrPortId",
        maximum=384,
        pattern=_FIELD_ID_RE,
    )
    if binding["nodeId"] not in node_ids:
        raise ValueError(f"{path}.binding.nodeId must reference a declared graph node.")
    if "mirrorBindings" in port:
        if direction != "input":
            raise ValueError(f"{path}.mirrorBindings is supported only for public inputs.")
        mirrors = _array(port["mirrorBindings"], f"{path}.mirrorBindings")
        if not mirrors:
            raise ValueError(f"{path}.mirrorBindings must be non-empty when present.")
        seen_bindings = {(binding["nodeId"], binding["fieldOrPortId"])}
        previous_key: tuple[str, str] | None = None
        for mirror_index, mirror_value in enumerate(mirrors):
            mirror_path = f"{path}.mirrorBindings[{mirror_index}]"
            mirror = _object(mirror_value, mirror_path)
            _exact_keys(mirror, mirror_path, required={"nodeId", "fieldOrPortId"})
            mirror_node_id = _string(
                mirror["nodeId"], f"{mirror_path}.nodeId", maximum=384, pattern=_ID_RE
            )
            mirror_field_id = _string(
                mirror["fieldOrPortId"],
                f"{mirror_path}.fieldOrPortId",
                maximum=384,
                pattern=_FIELD_ID_RE,
            )
            mirror_key = (mirror_node_id, mirror_field_id)
            if mirror_key in seen_bindings:
                raise ValueError(
                    f"{path}.mirrorBindings must not duplicate the primary or another mirror binding."
                )
            if previous_key is not None and mirror_key <= previous_key:
                raise ValueError(
                    f"{path}.mirrorBindings must be ordered canonically by nodeId and fieldOrPortId."
                )
            if mirror_node_id not in nodes_by_id:
                raise ValueError(f"{mirror_path}.nodeId must reference a declared graph node.")
            node_data = _object(nodes_by_id[mirror_node_id]["data"], f"{mirror_path} target data")
            params = _object(node_data.get("params"), f"{mirror_path} target params")
            param = _object(params.get(mirror_field_id), f"{mirror_path} target field")
            if param.get("display") == "output":
                raise ValueError(f"{mirror_path} cannot target an output field.")
            if not (_block_value_types_are_compatible_v2(port["valueType"], param.get("type"))
                    or _block_media_file_boundary_is_compatible_v2(port["valueType"], param)):
                raise ValueError(f"{mirror_path} targets an incompatible field type.")
            seen_bindings.add(mirror_key)
            previous_key = mirror_key
    return port_id


def _validate_boundary(
    boundary_value: Any,
    *,
    graph: dict[str, Any],
    source_kind: str,
) -> None:
    boundary = _object(boundary_value, "BlockDefinitionV2.boundary")
    _exact_keys(
        boundary,
        "BlockDefinitionV2.boundary",
        required={"mode", "inputs", "outputs"},
        optional={"derivation"},
    )
    mode = _string(boundary["mode"], "BlockDefinitionV2.boundary.mode")
    if mode not in {"explicit", "derived"}:
        raise ValueError("BlockDefinitionV2.boundary.mode is unsupported.")
    if source_kind in _CATALOG_SOURCE_KINDS and mode != "explicit":
        raise ValueError("Registered catalog BlockDefinitionV2 boundaries must be explicit.")
    if mode == "derived" and "derivation" not in boundary:
        raise ValueError("A derived BlockDefinitionV2 boundary requires derivation metadata.")
    if mode == "explicit" and "derivation" in boundary:
        raise ValueError("An explicit BlockDefinitionV2 boundary must not include derivation metadata.")

    public_port_ids: set[str] = set()
    for direction in ("inputs", "outputs"):
        values = _array(boundary[direction], f"BlockDefinitionV2.boundary.{direction}")
        port_ids: set[str] = set()
        for index, value in enumerate(values):
            port_id = _validate_port(
                value,
                f"BlockDefinitionV2.boundary.{direction}[{index}]",
                graph=graph,
                direction="input" if direction == "inputs" else "output",
            )
            if port_id in port_ids:
                raise ValueError(f"BlockDefinitionV2.boundary.{direction} contains duplicate portId {port_id!r}.")
            if port_id in public_port_ids:
                raise ValueError(
                    "BlockDefinitionV2.boundary public portIds must be unique across inputs and outputs; "
                    f"found {port_id!r} in both directions."
                )
            port_ids.add(port_id)
            public_port_ids.add(port_id)

    if "derivation" in boundary:
        derivation = _object(boundary["derivation"], "BlockDefinitionV2.boundary.derivation")
        _exact_keys(
            derivation,
            "BlockDefinitionV2.boundary.derivation",
            required={"algorithmVersion", "derivedAtDefinitionHash"},
        )
        _string(
            derivation["algorithmVersion"],
            "BlockDefinitionV2.boundary.derivation.algorithmVersion",
        )
        _string(
            derivation["derivedAtDefinitionHash"],
            "BlockDefinitionV2.boundary.derivation.derivedAtDefinitionHash",
            pattern=_HASH_RE,
            )


def _block_connection_types_v2(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    aliases = {
        "opaque": "any",
        "str": "string",
        "string": "string",
        "text": "string",
        "bool": "bool",
        "boolean": "bool",
        "float": "float",
        "double": "float",
        "number": "float",
        "dropdown": "string",
        "int": "int",
        "integer": "int",
    }
    normalized: set[str] = set()
    for item in values:
        candidate = item
        if isinstance(candidate, str):
            scalar_match = re.fullmatch(r"(?:builtins\.)?([a-z_][a-z\d_]*)", candidate.strip().lower())
            if scalar_match is not None:
                candidate = aliases.get(scalar_match.group(1), candidate)
        text = str(candidate if candidate is not None else "default").strip()
        if not text:
            text = "default"
        token = re.sub(r"[^\w-]", "_", text.rsplit(".", 1)[-1], flags=re.UNICODE).lower() or "default"
        if token not in {"default", "missing"}:
            normalized.add(token)
    return sorted(normalized)


def _block_value_types_are_compatible_v2(left: Any, right: Any) -> bool:
    left_types = _block_connection_types_v2(left)
    right_types = _block_connection_types_v2(right)
    compatible = (
        not left_types
        or not right_types
        or "any" in left_types
        or "any" in right_types
        or bool(set(left_types).intersection(right_types))
    )
    if compatible:
        return True
    # Mirror the client contract: one JSON integer can losslessly feed a
    # floating-point consumer (for example a generation frame rate mirrored
    # to a video-export FPS field), but not the other way around.
    return "int" in left_types and "float" in right_types


def _block_media_file_boundary_is_compatible_v2(value_type: Any, param: dict[str, Any]) -> bool:
    """Match media inputs to an explicitly matching file picker's path transport."""
    options = param.get("fieldOptions")
    if (param.get("display") != "filebrowser"
            or not _block_value_types_are_compatible_v2(param.get("type"), "string")
            or not isinstance(options, dict) or not isinstance(options.get("fileTypes"), list)):
        return False
    file_types = {value.strip().lower() for value in options["fileTypes"] if isinstance(value, str)}
    values = value_type if isinstance(value_type, list) else [value_type]
    patterns = {"image": r"images?|pil|pixels?", "video": r"videos?|frames?", "audio": r"audios?|sounds?|waveforms?"}
    return any(media in file_types and any(
        isinstance(value, str) and (re.search(rf"(?:^|[^a-z])(?:{pattern})(?:[^a-z]|$)", value, re.I)
            or (media == "video" and re.fullmatch(r"(?:typing\.)?(?:list|sequence)\[(?:PIL\.Image\.Image|image)\]", value.strip(), re.I)))
        for value in values
    ) for media, pattern in patterns.items())


def _validate_controls(
    controls_value: Any,
    *,
    graph: dict[str, Any],
) -> set[str]:
    controls = _array(controls_value, "BlockDefinitionV2.controls")
    nodes_by_id = {node["nodeId"]: node for node in graph["nodes"]}
    node_ids = set(nodes_by_id)
    control_ids: set[str] = set()
    for index, control_value in enumerate(controls):
        path = f"BlockDefinitionV2.controls[{index}]"
        control = _object(control_value, path)
        _exact_keys(
            control,
            path,
            required={"controlId", "label", "binding", "valueType", "order"},
            optional={"defaultValue", "required", "sealed", "group", "help", "mirrorBindings"},
        )
        control_id = _string(control["controlId"], f"{path}.controlId", maximum=384, pattern=_ID_RE)
        if control_id in control_ids:
            raise ValueError(f"BlockDefinitionV2.controls contains duplicate controlId {control_id!r}.")
        control_ids.add(control_id)
        _string(control["label"], f"{path}.label")
        _string(control["valueType"], f"{path}.valueType")
        order = _integer(control["order"], f"{path}.order")
        if order != index:
            raise ValueError("BlockDefinitionV2.controls must be ordered contiguously from zero.")
        binding = _object(control["binding"], f"{path}.binding")
        _exact_keys(binding, f"{path}.binding", required={"nodeId", "fieldId"})
        _string(binding["nodeId"], f"{path}.binding.nodeId", maximum=384, pattern=_ID_RE)
        _string(binding["fieldId"], f"{path}.binding.fieldId", maximum=384, pattern=_FIELD_ID_RE)
        if binding["nodeId"] not in node_ids:
            raise ValueError(f"{path}.binding.nodeId must reference a declared graph node.")
        if "mirrorBindings" in control:
            mirrors = _array(control["mirrorBindings"], f"{path}.mirrorBindings")
            if not mirrors:
                raise ValueError(f"{path}.mirrorBindings must be non-empty when present.")
            seen_bindings = {(binding["nodeId"], binding["fieldId"])}
            previous_key: tuple[str, str] | None = None
            for mirror_index, mirror_value in enumerate(mirrors):
                mirror_path = f"{path}.mirrorBindings[{mirror_index}]"
                mirror = _object(mirror_value, mirror_path)
                _exact_keys(mirror, mirror_path, required={"nodeId", "fieldId"})
                mirror_node_id = _string(
                    mirror["nodeId"], f"{mirror_path}.nodeId", maximum=384, pattern=_ID_RE
                )
                mirror_field_id = _string(
                    mirror["fieldId"], f"{mirror_path}.fieldId", maximum=384, pattern=_FIELD_ID_RE
                )
                mirror_key = (mirror_node_id, mirror_field_id)
                if mirror_key in seen_bindings:
                    raise ValueError(
                        f"{path}.mirrorBindings must not duplicate the primary or another mirror binding."
                    )
                if previous_key is not None and mirror_key <= previous_key:
                    raise ValueError(f"{path}.mirrorBindings must be ordered canonically by nodeId and fieldId.")
                if mirror_node_id not in nodes_by_id:
                    raise ValueError(f"{mirror_path}.nodeId must reference a declared graph node.")
                node_data = _object(nodes_by_id[mirror_node_id]["data"], f"{mirror_path} target data")
                params = _object(node_data.get("params"), f"{mirror_path} target params")
                param = _object(params.get(mirror_field_id), f"{mirror_path} target field")
                if param.get("display") == "output":
                    raise ValueError(f"{mirror_path} cannot target an output field.")
                if not _block_value_types_are_compatible_v2(
                    control["valueType"], param.get("type")
                ):
                    raise ValueError(f"{mirror_path} targets an incompatible field type.")
                seen_bindings.add(mirror_key)
                previous_key = mirror_key
        for key in ("required", "sealed"):
            if key in control:
                _bool(control[key], f"{path}.{key}")
        for key in ("group", "help"):
            if key in control:
                _string(control[key], f"{path}.{key}", maximum=4096 if key == "help" else 512)
        if "defaultValue" in control:
            _json_value(control["defaultValue"], f"{path}.defaultValue")
    return control_ids


def block_graph_subtree_node_ids_v2(graph: dict[str, Any], root_node_id: str) -> set[str]:
    """Resolve the same flat semantic subtree used by the client projector."""
    if not any(node["nodeId"] == root_node_id for node in graph["nodes"]):
        raise ValueError(f"Unknown Block V2 subtree {root_node_id}.")
    children: dict[str, list[str]] = {}
    for node_id, parent_id in block_graph_parent_ids_v2(graph).items():
        children.setdefault(parent_id, []).append(node_id)
    included: set[str] = set()

    def visit(node_id: str, ancestors: set[str]) -> None:
        if node_id in ancestors:
            raise ValueError(f"Cyclic Block V2 subtree at {node_id}.")
        if node_id in included:
            return
        included.add(node_id)
        for child in children.get(node_id, []):
            visit(child, ancestors | {node_id})

    visit(root_node_id, set())
    return included


def validate_block_container_interface_v1(
    value: Any, graph: dict[str, Any], owner_node_id: str
) -> BlockContainerInterfaceV1:
    """Strict local interface, hash-covered by its one existing semantic graph."""
    label = f"Block container interface V1 ({owner_node_id})"
    interface = _object(value, label)
    _exact_keys(interface, label, required={"schemaVersion", "boundary", "controls"}, optional={"previews"})
    if interface["schemaVersion"] != 1 or isinstance(interface["schemaVersion"], bool):
        raise ValueError(f"{label}.schemaVersion must be 1.")
    included = block_graph_subtree_node_ids_v2(graph, owner_node_id)
    scoped_graph = {**graph, "nodes": [node for node in graph["nodes"] if node["nodeId"] in included]}
    _validate_boundary(interface["boundary"], graph=scoped_graph, source_kind="user")
    boundary = interface["boundary"]
    if boundary["mode"] != "explicit":
        raise ValueError(f"{label} boundary must be explicit.")
    _validate_controls(interface["controls"], graph=scoped_graph)
    nodes_by_id = {node["nodeId"]: node for node in scoped_graph["nodes"]}

    def validate_field(entry: dict[str, Any], binding: dict[str, Any], direction: str) -> tuple[str, str]:
        field_id = binding.get("fieldId", binding.get("fieldOrPortId"))
        params = _object(nodes_by_id[binding["nodeId"]]["data"].get("params"), f"{label} target params")
        param = _object(params.get(field_id), f"{label} target field {binding['nodeId']}.{field_id}")
        if (direction == "output") != (param.get("display") == "output"):
            raise ValueError(f"{label} field {binding['nodeId']}.{field_id} has incompatible direction.")
        if not (_block_value_types_are_compatible_v2(entry["valueType"], param.get("type"))
                or (direction == "input" and _block_media_file_boundary_is_compatible_v2(entry["valueType"], param))):
            raise ValueError(f"{label} field {binding['nodeId']}.{field_id} has incompatible type.")
        return binding["nodeId"], field_id

    occupied_inputs: set[tuple[str, str]] = set()
    for direction in ("input", "output"):
        for port in boundary["inputs" if direction == "input" else "outputs"]:
            for binding in [port["binding"], *port.get("mirrorBindings", [])]:
                key = validate_field(port, binding, direction)
                if direction == "input":
                    if key in occupied_inputs:
                        raise ValueError(f"{label}: an internal input cannot be exposed by multiple local ports.")
                    occupied_inputs.add(key)
    occupied_controls: set[tuple[str, str]] = set()
    inputs_by_id = {port["portId"]: port for port in boundary["inputs"]}
    for control in interface["controls"]:
        if "defaultValue" in control:
            raise ValueError(f"{label}: control defaults belong to the bound fields, not this view.")
        for binding in [control["binding"], *control.get("mirrorBindings", [])]:
            key = validate_field(control, binding, "control")
            if key in occupied_controls:
                raise ValueError(f"{label}: a field cannot have multiple local controls.")
            occupied_controls.add(key)
        shared = inputs_by_id.get(control["controlId"])
        if shared:
            port_targets = {(b["nodeId"], b["fieldOrPortId"]) for b in [shared["binding"], *shared.get("mirrorBindings", [])]}
            control_targets = {(b["nodeId"], b["fieldId"]) for b in [control["binding"], *control.get("mirrorBindings", [])]}
            media_path_control = _block_value_types_are_compatible_v2(control["valueType"], "string") and all(
                _block_media_file_boundary_is_compatible_v2(shared["valueType"], nodes_by_id[node_id]["data"]["params"][field_id])
                for node_id, field_id in port_targets
            )
            if port_targets != control_targets or (shared["valueType"] != control["valueType"] and not media_path_control):
                raise ValueError(f"{label}: a shared input/control ID must bind the same complete field set and type.")
    if "previews" in interface:
        _validate_previews(interface["previews"], node_ids=set(nodes_by_id))
        media_tokens = {
            "image": r"images?|pil|pixels?", "video": r"videos?|frames?",
            "audio": r"audios?|sounds?|waveforms?", "text": r"texts?|strings?|str",
            "file": r"files?|paths?|uris?|urls?",
        }
        for preview in interface["previews"]:
            params = _object(nodes_by_id[preview["nodeId"]]["data"].get("params"), f"{label} preview params")
            field = _object(params.get(preview["outputPortId"]), f"{label} preview field")
            types = field.get("type")
            types = types if isinstance(types, list) else [types]
            media_type = preview["mediaType"]
            display = "ui_text" if media_type == "file" else f"ui_{media_type}"
            transported_preview = field.get("display") == display and (
                "type" not in field or any(
                    isinstance(entry, str) and re.fullmatch(r"url|uri|path|string|str|base64", entry, re.I)
                    for entry in types
                )
            )
            if field.get("display") not in {"output", display} or not (transported_preview or any(
                isinstance(entry, str) and re.search(rf"(?:^|[^a-z])(?:{media_tokens[media_type]})(?:[^a-z]|$)", entry, re.I)
                for entry in types
            )):
                raise ValueError(f"{label} preview {preview['nodeId']}.{preview['outputPortId']} has incompatible type or display.")
    return copy.deepcopy(interface)


def block_instance_preview_bindings_v2(definition: dict[str, Any], graph: dict[str, Any]) -> list[BlockPreviewBindingV2]:
    """Root bindings first, then unique local sources; one root-owned state per source."""
    node_ids = {node["nodeId"] for node in graph["nodes"]}
    bindings = copy.deepcopy([binding for binding in definition["previews"] if binding["nodeId"] in node_ids])
    by_source = {(binding["nodeId"], binding["outputPortId"]): binding for binding in bindings}
    for node in graph["nodes"]:
        for preview in node.get("containerInterface", {}).get("previews", []):
            key = (preview["nodeId"], preview["outputPortId"])
            existing = by_source.get(key)
            if existing is not None:
                if existing["mediaType"] != preview["mediaType"]:
                    raise ValueError(f"Block preview inventory V2: conflicting media types for {key[0]}.{key[1]}.")
                continue
            binding = copy.deepcopy(preview)
            binding.pop("primary", None)
            bindings.append(binding)
            by_source[key] = binding
    return bindings


def _validate_suggested_inputs(value: Any, *, control_ids: set[str]) -> None:
    suggestions = _array(value, "BlockDefinitionV2.suggestedInputs")
    suggestion_ids: set[str] = set()
    for index, suggestion_value in enumerate(suggestions):
        path = f"BlockDefinitionV2.suggestedInputs[{index}]"
        suggestion = _object(suggestion_value, path)
        _exact_keys(
            suggestion,
            path,
            required={"suggestionId", "label", "values"},
            optional={"source"},
        )
        suggestion_id = _string(
            suggestion["suggestionId"],
            f"{path}.suggestionId",
            maximum=384,
            pattern=_ID_RE,
        )
        if suggestion_id in suggestion_ids:
            raise ValueError(f"BlockDefinitionV2.suggestedInputs contains duplicate suggestionId {suggestion_id!r}.")
        suggestion_ids.add(suggestion_id)
        _string(suggestion["label"], f"{path}.label")
        if "source" in suggestion:
            _string(suggestion["source"], f"{path}.source", maximum=2048)
        values = _object(suggestion["values"], f"{path}.values")
        if any(not isinstance(key, str) for key in values):
            raise ValueError(f"{path}.values keys must be controlId strings.")
        unknown_controls = set(values) - control_ids
        if unknown_controls:
            raise ValueError(f"{path}.values references unknown controlId(s): {', '.join(sorted(unknown_controls))}.")
        _json_value(values, f"{path}.values")


def _validate_previews(value: Any, *, node_ids: set[str]) -> None:
    previews = _array(value, "BlockDefinitionV2.previews")
    primary_count = 0
    bindings: set[tuple[str, str]] = set()
    for index, preview_value in enumerate(previews):
        path = f"BlockDefinitionV2.previews[{index}]"
        preview = _object(preview_value, path)
        _exact_keys(
            preview,
            path,
            required={"nodeId", "outputPortId", "mediaType"},
            optional={"primary"},
        )
        _string(preview["nodeId"], f"{path}.nodeId", maximum=384, pattern=_ID_RE)
        _string(
            preview["outputPortId"],
            f"{path}.outputPortId",
            maximum=384,
            pattern=_FIELD_ID_RE,
        )
        if preview["nodeId"] not in node_ids:
            raise ValueError(f"{path}.nodeId must reference a declared graph node.")
        binding = (preview["nodeId"], preview["outputPortId"])
        if binding in bindings:
            raise ValueError("BlockDefinitionV2.previews must not contain duplicate bindings.")
        bindings.add(binding)
        media_type = _string(preview["mediaType"], f"{path}.mediaType")
        if media_type not in _MEDIA_TYPES:
            raise ValueError(f"{path}.mediaType is unsupported.")
        if "primary" in preview:
            _bool(preview["primary"], f"{path}.primary")
            primary_count += int(preview["primary"])
    if primary_count > 1:
        raise ValueError("BlockDefinitionV2.previews may contain at most one primary binding.")


def validate_block_definition_v2(payload: Any) -> BlockDefinitionV2:
    """Validate and clone one canonical ``BlockDefinitionV2``.

    The returned object has the same authoritative fields and values as the
    input.  Validation never derives ports, changes identifiers, fills
    defaults, recalculates hashes, or updates timestamps.
    """

    definition = _object(payload, "BlockDefinitionV2")
    _exact_keys(
        definition,
        "BlockDefinitionV2",
        required={
            "schemaVersion",
            "definitionId",
            "displayName",
            "contentHash",
            "source",
            "graph",
            "boundary",
            "controls",
            "previews",
            "ownership",
        },
        optional={"description", "suggestedInputs"},
    )
    if definition["schemaVersion"] != 2 or isinstance(definition["schemaVersion"], bool):
        raise ValueError("BlockDefinitionV2.schemaVersion must be 2.")
    _string(
        definition["definitionId"],
        "BlockDefinitionV2.definitionId",
        maximum=384,
        pattern=_ID_RE,
    )
    _string(definition["displayName"], "BlockDefinitionV2.displayName")
    _string(definition["contentHash"], "BlockDefinitionV2.contentHash", pattern=_HASH_RE)
    if "description" in definition:
        _string(definition["description"], "BlockDefinitionV2.description", maximum=4096)

    source = _validate_source(definition["source"])
    graph, node_ids = _validate_graph(definition["graph"])
    _validate_boundary(
        definition["boundary"],
        graph=graph,
        source_kind=source["kind"],
    )
    control_ids = _validate_controls(definition["controls"], graph=graph)
    inputs_by_id = {input_port["portId"]: input_port for input_port in definition["boundary"]["inputs"]}
    for control in definition["controls"]:
        shared_input = inputs_by_id.get(control["controlId"])
        if shared_input is None:
            continue
        input_binding = shared_input["binding"]
        control_binding = control["binding"]
        if (
            input_binding["nodeId"] != control_binding["nodeId"]
            or input_binding["fieldOrPortId"] != control_binding["fieldId"]
        ):
            raise ValueError(
                f"BlockDefinitionV2 input and control {control['controlId']!r} share an id "
                "but bind different graph fields."
            )
    if "suggestedInputs" in definition:
        _validate_suggested_inputs(definition["suggestedInputs"], control_ids=control_ids)
    _validate_previews(definition["previews"], node_ids=node_ids)
    block_instance_preview_bindings_v2(definition, graph)

    ownership = _object(definition["ownership"], "BlockDefinitionV2.ownership")
    _exact_keys(
        ownership,
        "BlockDefinitionV2.ownership",
        required={"kind", "definitionMutable"},
    )
    ownership_kind = _string(ownership["kind"], "BlockDefinitionV2.ownership.kind")
    if ownership_kind not in {"registered", "user"}:
        raise ValueError("BlockDefinitionV2.ownership.kind is unsupported.")
    _bool(ownership["definitionMutable"], "BlockDefinitionV2.ownership.definitionMutable")
    if source["kind"] in _CATALOG_SOURCE_KINDS:
        if ownership != {"kind": "registered", "definitionMutable": False}:
            raise ValueError("Registered catalog BlockDefinitionV2 ownership must be registered and immutable.")
    elif ownership != {"kind": "user", "definitionMutable": True}:
        raise ValueError("User and Hub-import BlockDefinitionV2 ownership must be mutable and user-owned.")

    _json_value(definition, "BlockDefinitionV2")
    expected_hash = block_definition_content_hash_v2(definition)
    if definition["contentHash"] != expected_hash:
        raise ValueError(f"BlockDefinitionV2.contentHash must be {expected_hash}.")
    encoded = _stable_json(definition).encode("utf-8")
    if len(encoded) > _MAX_DEFINITION_BYTES:
        raise ValueError(f"BlockDefinitionV2 exceeds the {_MAX_DEFINITION_BYTES}-byte persistence limit.")
    return copy.deepcopy(definition)


def _validate_instance_interface(
    value: Any,
    *,
    definition: BlockDefinitionV2,
    effective_graph: dict[str, Any],
) -> BlockEffectiveInterfaceV2:
    interface = _object(value, "BlockInstanceV2.effectiveInterface")
    _exact_keys(
        interface,
        "BlockInstanceV2.effectiveInterface",
        required={"boundary", "controls", "baseInterfaceHash", "effectiveInterfaceHash"},
    )
    _validate_boundary(
        interface["boundary"],
        graph=effective_graph,
        source_kind=definition["source"]["kind"],
    )
    _validate_controls(interface["controls"], graph=effective_graph)
    inputs_by_id = {
        input_port["portId"]: input_port for input_port in interface["boundary"]["inputs"]
    }
    for control in interface["controls"]:
        shared_input = inputs_by_id.get(control["controlId"])
        if shared_input is None:
            continue
        input_binding = shared_input["binding"]
        control_binding = control["binding"]
        if (
            input_binding["nodeId"] != control_binding["nodeId"]
            or input_binding["fieldOrPortId"] != control_binding["fieldId"]
        ):
            raise ValueError(
                f"BlockInstanceV2 input and control {control['controlId']!r} share an id "
                "but bind different graph fields."
            )

    _string(
        interface["baseInterfaceHash"],
        "BlockInstanceV2.effectiveInterface.baseInterfaceHash",
        pattern=_HASH_RE,
    )
    _string(
        interface["effectiveInterfaceHash"],
        "BlockInstanceV2.effectiveInterface.effectiveInterfaceHash",
        pattern=_HASH_RE,
    )
    base_hash = block_interface_hash_v2(
        {"boundary": definition["boundary"], "controls": definition["controls"]}
    )
    if interface["baseInterfaceHash"] != base_hash:
        raise ValueError(
            f"BlockInstanceV2.effectiveInterface.baseInterfaceHash must be {base_hash}."
        )
    effective_hash = block_interface_hash_v2(interface)
    if interface["effectiveInterfaceHash"] != effective_hash:
        raise ValueError(
            "BlockInstanceV2.effectiveInterface.effectiveInterfaceHash must be "
            f"{effective_hash}."
        )
    return copy.deepcopy(interface)


_INSTANCE_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z$"
)


def _instance_timestamp(value: Any, path: str) -> tuple[str, datetime]:
    text = _string(value, path, maximum=64)
    if _INSTANCE_TIMESTAMP_RE.fullmatch(text) is None:
        raise ValueError(f"{path} must be an ISO-8601 UTC timestamp.")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{path} must be an ISO-8601 UTC timestamp.") from error
    return text, parsed


def _validate_instance_preview_states(
    value: Any,
    *,
    definition: BlockDefinitionV2,
    effective_graph: BlockGraphV2,
) -> list[dict[str, Any]]:
    states = _array(value, "BlockInstanceV2.previewStates")
    expected = block_instance_preview_bindings_v2(definition, effective_graph)
    if len(states) != len(expected):
        raise ValueError(
            "BlockInstanceV2.previewStates must preserve the ordered definition and local preview bindings."
        )
    definition_node_ids = {node["nodeId"] for node in [*definition["graph"]["nodes"], *effective_graph["nodes"]]}
    for index, state_value in enumerate(states):
        path = f"BlockInstanceV2.previewStates[{index}]"
        state = _object(state_value, path)
        _exact_keys(
            state,
            path,
            required={"binding"},
            optional={"mediaReference", "taskId", "status"},
        )
        _validate_previews([state["binding"]], node_ids=definition_node_ids)
        if _stable_json(state["binding"]) != _stable_json(expected[index]):
            raise ValueError(
                "BlockInstanceV2.previewStates must preserve the ordered definition and local preview bindings."
            )
        if "mediaReference" in state:
            _string(state["mediaReference"], f"{path}.mediaReference", maximum=8192)
        if "taskId" in state:
            _string(state["taskId"], f"{path}.taskId")
        if "status" in state:
            status = _string(state["status"], f"{path}.status")
            if status not in {"idle", "queued", "running", "complete", "failed"}:
                raise ValueError(f"{path}.status is unsupported.")
    return copy.deepcopy(states)


def _validate_instance_authorities(
    value: Any,
    *,
    definition_ref: dict[str, Any],
    effective_graph_hash: str,
) -> list[dict[str, Any]]:
    authorities = _array(value, "BlockInstanceV2.authorities")
    seen_kinds: set[str] = set()
    for index, authority_value in enumerate(authorities):
        path = f"BlockInstanceV2.authorities[{index}]"
        authority = _object(authority_value, path)
        _exact_keys(
            authority,
            path,
            required={
                "kind",
                "definitionId",
                "definitionContentHash",
                "effectiveGraphHash",
                "executionParameterHash",
                "artifactRevisions",
                "admissionId",
                "issuedAt",
            },
            optional={"expiresAt"},
        )
        kind = _string(authority["kind"], f"{path}.kind")
        if kind not in {"reviewed_execution", "auto", "publication"}:
            raise ValueError(f"{path}.kind is unsupported.")
        if kind in seen_kinds:
            raise ValueError("BlockInstanceV2.authorities must not contain duplicate kinds.")
        seen_kinds.add(kind)
        _string(authority["definitionId"], f"{path}.definitionId", maximum=384, pattern=_ID_RE)
        for key in ("definitionContentHash", "effectiveGraphHash", "executionParameterHash"):
            _string(authority[key], f"{path}.{key}", pattern=_HASH_RE)
        _string(authority["admissionId"], f"{path}.admissionId")
        _, issued_at = _instance_timestamp(authority["issuedAt"], f"{path}.issuedAt")
        if "expiresAt" in authority:
            _, expires_at = _instance_timestamp(authority["expiresAt"], f"{path}.expiresAt")
            if expires_at <= issued_at:
                raise ValueError(f"{path}.expiresAt must follow issuedAt.")
        revisions = _object(authority["artifactRevisions"], f"{path}.artifactRevisions")
        for repository, revision in revisions.items():
            _string(repository, f"{path}.artifactRevisions repository", maximum=384, pattern=_REPOSITORY_ID_RE)
            _string(
                revision,
                f"{path}.artifactRevisions.{repository}",
                maximum=40,
                pattern=_IMMUTABLE_REVISION_RE,
            )
        if (
            authority["definitionId"] != definition_ref["definitionId"]
            or authority["definitionContentHash"] != definition_ref["contentHash"]
            or authority["effectiveGraphHash"] != effective_graph_hash
        ):
            raise ValueError(f"{path} does not describe the current definition and effective graph.")
    return copy.deepcopy(authorities)


def _validate_route_selection_v1(value: Any) -> BlockRouteSelectionV1:
    """Validate bounded inactive routes without admitting recursive instances."""

    selection = _object(value, "BlockRouteSelectionV1")
    _exact_keys(
        selection,
        "BlockRouteSelectionV1",
        required={"schemaVersion", "routeSetId", "selectedRouteKey", "inactiveDrafts"},
    )
    if selection["schemaVersion"] != 1 or isinstance(selection["schemaVersion"], bool):
        raise ValueError("BlockRouteSelectionV1.schemaVersion must be 1.")
    route_set_id = _string(
        selection["routeSetId"],
        "BlockRouteSelectionV1.routeSetId",
        maximum=384,
        pattern=_ID_RE,
    )
    selected_route_key = _string(
        selection["selectedRouteKey"],
        "BlockRouteSelectionV1.selectedRouteKey",
        maximum=384,
        pattern=_ID_RE,
    )
    raw_drafts = _object(selection["inactiveDrafts"], "BlockRouteSelectionV1.inactiveDrafts")
    if len(raw_drafts) > 8:
        raise ValueError("BlockRouteSelectionV1.inactiveDrafts may contain at most 8 exact-route drafts.")
    if selected_route_key in raw_drafts:
        raise ValueError("BlockRouteSelectionV1.inactiveDrafts must not contain the selected active route.")
    normalized_drafts: dict[str, BlockRouteDraftV1] = {}
    for draft_key, draft_value in raw_drafts.items():
        _string(draft_key, "BlockRouteSelectionV1 draft key", maximum=384, pattern=_ID_RE)
        path = f"BlockRouteDraftV1.{draft_key}"
        draft = _object(draft_value, path)
        _exact_keys(
            draft,
            path,
            required={
                "schemaVersion",
                "routeKey",
                "definitionRef",
                "definitionSnapshot",
                "effectiveGraph",
                "effectiveInterface",
                "values",
                "customization",
                "internalLayout",
            },
            optional={"internalLayoutMode", "collapsedContainerNodeIds"},
        )
        if draft["schemaVersion"] != 1 or isinstance(draft["schemaVersion"], bool):
            raise ValueError(f"{path}.schemaVersion must be 1.")
        if draft["routeKey"] != draft_key:
            raise ValueError(f"{path}.routeKey must match its inactiveDrafts key.")
        definition = validate_block_definition_v2(draft["definitionSnapshot"])
        if (
            definition["ownership"] != {"kind": "registered", "definitionMutable": False}
            or definition["source"]["kind"] not in _CATALOG_SOURCE_KINDS
        ):
            raise ValueError(f"{path} must contain one immutable registered definition.")
        validated = validate_block_instance_v2(
            {
                "schemaVersion": 2,
                "instanceId": draft_key,
                "definitionRef": draft["definitionRef"],
                "definitionSnapshot": definition,
                "effectiveGraph": draft["effectiveGraph"],
                "effectiveInterface": draft["effectiveInterface"],
                "values": draft["values"],
                "customization": draft["customization"],
                "presentation": {
                    "expanded": False,
                    "position": {"x": 0, "y": 0},
                    "size": {"width": 1, "height": 1},
                    "internalLayout": draft["internalLayout"],
                    **(
                        {"internalLayoutMode": draft["internalLayoutMode"]}
                        if "internalLayoutMode" in draft
                        else {}
                    ),
                    **(
                        {"collapsedContainerNodeIds": draft["collapsedContainerNodeIds"]}
                        if "collapsedContainerNodeIds" in draft
                        else {}
                    ),
                },
                "previewStates": [
                    {"binding": preview, "status": "idle"}
                    for preview in block_instance_preview_bindings_v2(definition, _validate_graph(draft["effectiveGraph"])[0])
                ],
                "authorities": [],
            }
        )
        normalized_drafts[draft_key] = {
            "schemaVersion": 1,
            "routeKey": draft_key,
            "definitionRef": validated["definitionRef"],
            "definitionSnapshot": validated["definitionSnapshot"],
            "effectiveGraph": validated["effectiveGraph"],
            "effectiveInterface": validated["effectiveInterface"],
            "values": validated["values"],
            "customization": validated["customization"],
            "internalLayout": validated["presentation"]["internalLayout"],
            **(
                {"internalLayoutMode": validated["presentation"]["internalLayoutMode"]}
                if "internalLayoutMode" in validated["presentation"]
                else {}
            ),
            **(
                {"collapsedContainerNodeIds": validated["presentation"]["collapsedContainerNodeIds"]}
                if "collapsedContainerNodeIds" in validated["presentation"]
                else {}
            ),
        }
    return {
        "schemaVersion": 1,
        "routeSetId": route_set_id,
        "selectedRouteKey": selected_route_key,
        "inactiveDrafts": normalized_drafts,
    }


def validate_block_instance_v2(payload: Any) -> BlockInstanceV2:
    """Strictly validate and normalize one workflow-owned ``BlockInstanceV2``.

    ``effectiveInterface`` omission is the sole admitted legacy shape and is
    deterministically restored from the embedded definition. Unknown fields,
    stale hashes, nested composites, invalid copy-on-write state, and
    non-finite values all fail closed exactly as they do in the client.
    """

    instance = _object(payload, "BlockInstanceV2")
    _exact_keys(
        instance,
        "BlockInstanceV2",
        required={
            "schemaVersion",
            "instanceId",
            "definitionRef",
            "definitionSnapshot",
            "effectiveGraph",
            "values",
            "customization",
            "presentation",
            "previewStates",
            "authorities",
        },
        optional={"effectiveInterface", "routeSelection"},
    )
    if instance["schemaVersion"] != 2 or isinstance(instance["schemaVersion"], bool):
        raise ValueError("BlockInstanceV2.schemaVersion must be 2.")
    _string(instance["instanceId"], "BlockInstanceV2.instanceId", maximum=384, pattern=_ID_RE)

    definition_ref = _object(instance["definitionRef"], "BlockInstanceV2.definitionRef")
    _exact_keys(
        definition_ref,
        "BlockInstanceV2.definitionRef",
        required={"definitionId", "contentHash"},
    )
    _string(
        definition_ref["definitionId"],
        "BlockInstanceV2.definitionRef.definitionId",
        maximum=384,
        pattern=_ID_RE,
    )
    _string(
        definition_ref["contentHash"],
        "BlockInstanceV2.definitionRef.contentHash",
        pattern=_HASH_RE,
    )
    definition = validate_block_definition_v2(instance["definitionSnapshot"])
    if (
        definition_ref["definitionId"] != definition["definitionId"]
        or definition_ref["contentHash"] != definition["contentHash"]
    ):
        raise ValueError(
            "BlockInstanceV2.definitionRef must match the embedded definition snapshot."
        )

    effective_graph, effective_node_ids = _validate_graph(instance["effectiveGraph"])
    base_interface_hash = block_interface_hash_v2(
        {"boundary": definition["boundary"], "controls": definition["controls"]}
    )
    raw_interface = instance.get(
        "effectiveInterface",
        {
            "boundary": definition["boundary"],
            "controls": definition["controls"],
            "baseInterfaceHash": base_interface_hash,
            "effectiveInterfaceHash": base_interface_hash,
        },
    )
    effective_interface = _validate_instance_interface(
        raw_interface,
        definition=definition,
        effective_graph=effective_graph,
    )

    values = _object(instance["values"], "BlockInstanceV2.values")
    if any(not isinstance(key, str) for key in values):
        raise ValueError("BlockInstanceV2.values keys must be input/control IDs.")
    value_ids = {
        *(control["controlId"] for control in effective_interface["controls"]),
        *(port["portId"] for port in effective_interface["boundary"]["inputs"]),
    }
    unknown_value_ids = set(values) - value_ids
    if unknown_value_ids:
        raise ValueError(
            "BlockInstanceV2.values contains unknown input/control ID(s): "
            f"{', '.join(sorted(unknown_value_ids))}."
        )
    _json_value(values, "BlockInstanceV2.values")

    customization = _object(instance["customization"], "BlockInstanceV2.customization")
    _exact_keys(
        customization,
        "BlockInstanceV2.customization",
        required={"state", "baseGraphHash", "effectiveGraphHash"},
    )
    state = _string(customization["state"], "BlockInstanceV2.customization.state")
    if state not in {"unchanged", "parameters_changed", "structure_changed"}:
        raise ValueError("BlockInstanceV2.customization.state is unsupported.")
    for key in ("baseGraphHash", "effectiveGraphHash"):
        _string(customization[key], f"BlockInstanceV2.customization.{key}", pattern=_HASH_RE)
    if customization["baseGraphHash"] != definition["graph"]["graphHash"]:
        raise ValueError(
            "BlockInstanceV2.customization.baseGraphHash does not match the definition graph."
        )
    if customization["effectiveGraphHash"] != effective_graph["graphHash"]:
        raise ValueError(
            "BlockInstanceV2.customization.effectiveGraphHash does not match the effective graph."
        )
    structure_changed = (
        effective_graph["graphHash"] != definition["graph"]["graphHash"]
        or effective_interface["effectiveInterfaceHash"]
        != effective_interface["baseInterfaceHash"]
    )
    if state != "structure_changed" and structure_changed:
        raise ValueError(
            "BlockInstanceV2.customization only structure_changed may use a changed graph or interface."
        )
    if state == "structure_changed" and not structure_changed:
        raise ValueError(
            "BlockInstanceV2.customization structure_changed requires a different graph or interface."
        )

    presentation = _object(instance["presentation"], "BlockInstanceV2.presentation")
    _exact_keys(
        presentation,
        "BlockInstanceV2.presentation",
        required={"expanded", "position", "size", "internalLayout"},
        optional={"internalLayoutMode", "collapsedContainerNodeIds"},
    )
    _bool(presentation["expanded"], "BlockInstanceV2.presentation.expanded")
    position = _object(presentation["position"], "BlockInstanceV2.presentation.position")
    _exact_keys(position, "BlockInstanceV2.presentation.position", required={"x", "y"})
    _finite_number(position["x"], "BlockInstanceV2.presentation.position.x")
    _finite_number(position["y"], "BlockInstanceV2.presentation.position.y")
    size = _object(presentation["size"], "BlockInstanceV2.presentation.size")
    _exact_keys(size, "BlockInstanceV2.presentation.size", required={"width", "height"})
    _finite_number(size["width"], "BlockInstanceV2.presentation.size.width", minimum=1)
    _finite_number(size["height"], "BlockInstanceV2.presentation.size.height", minimum=1)
    if "internalLayoutMode" in presentation and presentation["internalLayoutMode"] not in {
        "root",
        "hierarchical",
    }:
        raise ValueError(
            "BlockInstanceV2.presentation.internalLayoutMode must be root or hierarchical."
        )
    if "collapsedContainerNodeIds" in presentation:
        collapsed_ids = presentation["collapsedContainerNodeIds"]
        if (
            not isinstance(collapsed_ids, list)
            or len(collapsed_ids) > len(effective_graph["nodes"])
            or len(collapsed_ids) != len(set(collapsed_ids))
        ):
            raise ValueError(
                "BlockInstanceV2.presentation.collapsedContainerNodeIds must be a bounded unique array."
            )
        modular_container_ids = set(modular_container_node_ids_v2(effective_graph))
        for index, node_id in enumerate(collapsed_ids):
            _string(
                node_id,
                f"BlockInstanceV2.presentation.collapsedContainerNodeIds.{index}",
                maximum=384,
                pattern=_ID_RE,
            )
            if node_id not in modular_container_ids:
                raise ValueError(
                    "BlockInstanceV2.presentation.collapsedContainerNodeIds references "
                    f"non-container node {node_id}."
                )
    internal_layout = _object(
        presentation["internalLayout"],
        "BlockInstanceV2.presentation.internalLayout",
    )
    unknown_layout_ids = set(internal_layout) - effective_node_ids
    if unknown_layout_ids:
        raise ValueError(
            "BlockInstanceV2.presentation.internalLayout references unknown node ID(s): "
            f"{', '.join(sorted(unknown_layout_ids))}."
        )
    for node_id, layout_value in internal_layout.items():
        path = f"BlockInstanceV2.presentation.internalLayout.{node_id}"
        layout = _object(layout_value, path)
        _exact_keys(layout, path, required={"x", "y"}, optional={"width", "height"})
        _finite_number(layout["x"], f"{path}.x")
        _finite_number(layout["y"], f"{path}.y")
        for key in ("width", "height"):
            if key in layout:
                _finite_number(layout[key], f"{path}.{key}", minimum=1)

    preview_states = _validate_instance_preview_states(
        instance["previewStates"],
        definition=definition,
        effective_graph=effective_graph,
    )
    authorities = _validate_instance_authorities(
        instance["authorities"],
        definition_ref=definition_ref,
        effective_graph_hash=effective_graph["graphHash"],
    )
    route_selection = (
        _validate_route_selection_v1(instance["routeSelection"])
        if "routeSelection" in instance
        else None
    )
    normalized = {
        **copy.deepcopy(instance),
        "definitionSnapshot": definition,
        "effectiveGraph": copy.deepcopy(effective_graph),
        "effectiveInterface": effective_interface,
        "values": copy.deepcopy(values),
        "previewStates": preview_states,
        "authorities": authorities,
        **({"routeSelection": route_selection} if route_selection is not None else {}),
    }
    _json_value(normalized, "BlockInstanceV2")
    encoded = _stable_json(normalized).encode("utf-8")
    if len(encoded) > _MAX_INSTANCE_BYTES:
        raise ValueError(f"BlockInstanceV2 exceeds the {_MAX_INSTANCE_BYTES}-byte persistence limit.")
    return normalized  # type: ignore[return-value]


def validate_user_store_block_definition_v2(payload: Any) -> BlockDefinitionV2:
    """Validate a V2 definition accepted by the user-owned Studio block store."""

    definition = validate_block_definition_v2(payload)
    if definition["source"]["kind"] not in _USER_STORE_SOURCE_KINDS:
        raise ValueError("Registered catalog definitions cannot be saved or overwritten through /studio/blocks.")
    if definition["ownership"]["kind"] != "user":
        raise ValueError("Only user-owned BlockDefinitionV2 definitions can be saved through /studio/blocks.")
    return definition
