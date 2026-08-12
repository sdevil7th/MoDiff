"""Generic task-template contracts derived from reviewed Studio execution specs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping


TASK_TEMPLATE_CONTRACT_SCHEMA_VERSION = 1
TASK_TEMPLATE_CONTRACT_CANONICALIZATION_VERSION = 1

_MEDIA_REQUIREMENT_KEYS = {
    "requiredImages": "image",
    "requiredVideos": "video",
    "requiredAudio": "audio",
}
_MEDIA_FIELDS = {
    "referenceImages": ("image", ("loadImage",)),
    "lastImage": ("image", ("loadLastImage",)),
    "maskImage": ("image", ("loadMask",)),
    "controlImage": ("image", ("loadControlImage", "loadImage")),
    "sourceVideo": ("video", ("loadVideo",)),
    "maskVideo": ("video", ("loadMaskVideo",)),
    "controlVideo": ("video", ("loadControlVideo",)),
    "poseVideo": ("video", ("loadPoseVideo",)),
    "faceVideo": ("video", ("loadFaceVideo",)),
    "backgroundVideo": ("video", ("loadBackgroundVideo",)),
    "sourceAudio": ("audio", ("loadAudio",)),
    "referenceAudio": ("audio", ("loadReferenceAudio",)),
}
_OUTPUT_NODE_KEYS = {
    "image": ("modules.Image.Preview",),
    "video": ("modules.Video.Export", "modules.Video.ExportWithAudio"),
    "audio": ("modules.Audio.Export",),
    "json": ("modules.Primitive.DataViewer",),
}
_OUTPUT_INPUT_HANDLES = {"image": "image", "video": "video", "audio": "audio", "json": "data"}


class TaskTemplateContractError(ValueError):
    """A task-template or checked-in graph contract is incomplete or ambiguous."""


def _ordered_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _ordered_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_ordered_value(item) for item in value]
    return value


def _stable_json(value: Any) -> str:
    import json

    return json.dumps(_ordered_value(value), ensure_ascii=False, separators=(",", ":"))


def _hash_string(value: str) -> str:
    digest = 0x811C9DC5
    encoded = value.encode("utf-16-le")
    for index in range(0, len(encoded), 2):
        digest ^= encoded[index] | encoded[index + 1] << 8
        digest = digest * 0x01000193 & 0xFFFFFFFF
    return f"{digest:08x}"


def _required_media(input_contract: Any, specification: Mapping[str, Any]) -> list[dict[str, Any]]:
    if input_contract is None:
        return []
    if not isinstance(input_contract, Mapping):
        raise TaskTemplateContractError("Task-template input contract must be an object.")
    required = []
    seen = set()
    for contract_key, kind in _MEDIA_REQUIREMENT_KEYS.items():
        values = input_contract.get(contract_key, [])
        if not isinstance(values, list):
            raise TaskTemplateContractError(f"Task-template {contract_key} must be a list.")
        for field in values:
            field_contract = _MEDIA_FIELDS.get(field) if isinstance(field, str) else None
            if field_contract is None or field_contract[0] != kind or field in seen:
                raise TaskTemplateContractError(f"Task-template media field {field!r} is invalid or duplicated.")
            seen.add(field)
            required.append({"kind": kind, "field": field, "minimumCount": 1})
    # The reviewed execution spec is the final authority for graph inputs. A
    # few older capabilities predate schema-v2 inputContracts, so normalize
    # their generic bindings instead of adding model-name exceptions.
    for binding in specification.get("bindings", []):
        if not isinstance(binding, (list, tuple)) or len(binding) != 3:
            continue
        field = binding[2]
        field_contract = _MEDIA_FIELDS.get(field) if isinstance(field, str) else None
        if field_contract is not None and field not in seen:
            seen.add(field)
            required.append({"kind": field_contract[0], "field": field, "minimumCount": 1})
    return required


def _output_contract(specification: Mapping[str, Any], media_kind: str) -> dict[str, str]:
    expected_node_keys = _OUTPUT_NODE_KEYS.get(media_kind)
    roles = specification.get("roles")
    edges = specification.get("edges")
    if expected_node_keys is None or not isinstance(roles, (list, tuple)) or not isinstance(edges, (list, tuple)):
        raise TaskTemplateContractError("Task-template output contract is invalid.")
    outgoing = {edge[0] for edge in edges if isinstance(edge, (list, tuple)) and len(edge) == 4}
    sinks = [
        item
        for item in roles
        if isinstance(item, (list, tuple)) and len(item) == 4 and item[0] not in outgoing
    ]
    if len(sinks) != 1 or sinks[0][1] not in expected_node_keys:
        raise TaskTemplateContractError(
            f"Task-template {media_kind!r} graph must end at exactly one reviewed output node."
        )
    role = sinks[0][0]
    incoming = [
        edge for edge in edges if isinstance(edge, (list, tuple)) and len(edge) == 4 and edge[2] == role
    ]
    media_inputs = [edge for edge in incoming if edge[3] == _OUTPUT_INPUT_HANDLES[media_kind]]
    if len(media_inputs) != 1:
        raise TaskTemplateContractError("Task-template output node must have exactly one reviewed media input edge.")
    return {
        "mediaKind": media_kind,
        "role": role,
        "nodeKey": sinks[0][1],
        "inputHandle": media_inputs[0][3],
    }


def _mode_qualification(capability: Mapping[str, Any], mode: str) -> str:
    qualified_modes = capability.get("qualifiedModes")
    status = capability.get("qualificationStatus") or "graph-qualified"
    if isinstance(qualified_modes, list):
        if mode in qualified_modes:
            return "qualified"
        if status == "qualified":
            return "graph-qualified-execution-pending"
    return str(status)


def build_task_template_contracts(
    capabilities: Iterable[Mapping[str, Any]],
    execution_specs: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build one exact, non-executable template descriptor per execution spec."""

    capability_by_model = {}
    for capability in capabilities:
        model_type = capability.get("modelType")
        if not isinstance(model_type, str) or model_type in capability_by_model:
            raise TaskTemplateContractError("Task-template capabilities must have unique model types.")
        capability_by_model[model_type] = capability

    contracts = []
    ids = set()
    pairs = set()
    for specification in execution_specs:
        model_type = specification.get("modelType")
        mode = specification.get("mode")
        capability = capability_by_model.get(model_type)
        if not isinstance(model_type, str) or not isinstance(mode, str) or capability is None:
            raise TaskTemplateContractError("Task-template execution spec has no exact capability.")
        pair = (model_type, mode)
        contract_id = f"task-template:{specification.get('id')}"
        if pair in pairs or contract_id in ids:
            raise TaskTemplateContractError("Task-template contracts must have unique ids and model/task pairs.")
        pairs.add(pair)
        ids.add(contract_id)

        profiles = capability.get("executionProfiles")
        matches = [
            profile
            for profile in profiles if isinstance(profile, Mapping)
        ] if isinstance(profiles, list) else []
        matches = [profile for profile in matches if profile.get("id") == specification.get("executionProfileId")]
        if len(matches) != 1:
            raise TaskTemplateContractError("Task-template execution profile is missing or ambiguous.")
        profile = matches[0]
        identity = (
            (profile.get("loader_module"), specification.get("loaderModule")),
            (profile.get("loader_action"), specification.get("loaderAction")),
            (profile.get("pipeline_class"), specification.get("pipelineClass")),
            (profile.get("default_repo"), specification.get("defaultRepo")),
        )
        if any(left != right or not isinstance(left, str) or not left for left, right in identity):
            raise TaskTemplateContractError("Task-template loader identity does not match its execution profile.")
        loader_key = f"{profile['loader_module']}.{profile['loader_action']}"
        loader_roles = [
            item[0]
            for item in specification.get("roles", [])
            if isinstance(item, (list, tuple)) and len(item) == 4 and item[1] == loader_key
        ]
        if len(loader_roles) != 1:
            raise TaskTemplateContractError("Task-template graph must contain its exact loader once.")
        loader_repositories = list(
            dict.fromkeys(
                repository
                for repository in (
                    profile["default_repo"],
                    profile.get("fallback_repo"),
                    *(profile.get("compatible_repos") or []),
                )
                if isinstance(repository, str) and repository
            )
        )

        media_kind = capability.get("mediaKind") or capability.get("outputKind") or "image"
        input_contracts = capability.get("inputContracts") or capability.get("modeRequirements") or {}
        if not isinstance(input_contracts, Mapping):
            raise TaskTemplateContractError("Task-template capability input contracts are invalid.")
        semantic = {
            "schemaVersion": TASK_TEMPLATE_CONTRACT_SCHEMA_VERSION,
            "canonicalizationVersion": TASK_TEMPLATE_CONTRACT_CANONICALIZATION_VERSION,
            "id": contract_id,
            "modelType": model_type,
            "mode": mode,
            "mediaKind": media_kind,
            "executionProfileId": specification["executionProfileId"],
            "executionSpecId": specification["id"],
            "executionSpecContentHash": specification["contentHash"],
            "loaderModule": specification["loaderModule"],
            "loaderAction": specification["loaderAction"],
            "loaderRole": loader_roles[0],
            "pipelineClass": specification["pipelineClass"],
            "defaultRepo": specification["defaultRepo"],
            "loaderRepositories": loader_repositories,
            "requiredMedia": _required_media(input_contracts.get(mode, {}), specification),
            "output": _output_contract(specification, str(media_kind)),
            "qualificationStatus": _mode_qualification(capability, mode),
            # P2 graph skeletons remain planning-only until the separate remote
            # qualification/review slice explicitly opts an exact pair in.
            "galleryEligible": False,
        }
        semantic["contentHash"] = f"task-template-v1-{_hash_string(_stable_json(semantic))}"
        contracts.append(semantic)
    return sorted(contracts, key=lambda item: item["id"])


def _field_value(field: Any) -> Any:
    if not isinstance(field, Mapping):
        return None
    value = field.get("value")
    if isinstance(value, Mapping):
        return value.get("value")
    return value


def validate_task_template_graph(
    graph: Mapping[str, Any],
    workflow: Mapping[str, Any],
    contract: Mapping[str, Any],
    specification: Mapping[str, Any],
) -> None:
    """Validate one checked-in graph against its exact generic task contract."""

    if workflow.get("modelType") != contract.get("modelType") or workflow.get("mode") != contract.get("mode"):
        raise TaskTemplateContractError("Workflow identity does not match its task-template contract.")
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise TaskTemplateContractError("Task-template graph must contain node and edge lists.")
    nodes_by_role: dict[str, Mapping[str, Any]] = {}
    ids = set()
    for node in nodes:
        if not isinstance(node, Mapping) or not isinstance(node.get("id"), str) or node["id"] in ids:
            raise TaskTemplateContractError("Task-template graph node ids must be unique strings.")
        ids.add(node["id"])
        data = node.get("data")
        role = data.get("studioRole") if isinstance(data, Mapping) else None
        if isinstance(role, str):
            if role in nodes_by_role:
                raise TaskTemplateContractError("Task-template graph roles must be unique.")
            nodes_by_role[role] = node

    for role, node_key, _x, _y in specification["roles"]:
        node = nodes_by_role.get(role)
        data = node.get("data") if isinstance(node, Mapping) else None
        if not isinstance(data, Mapping) or f"{data.get('module')}.{data.get('action')}" != node_key:
            raise TaskTemplateContractError("Task-template graph node identity does not match its execution spec.")

    loader = nodes_by_role[contract["loaderRole"]]
    params = loader["data"].get("params", {})
    binding_sources = {
        source: (role, param)
        for role, param, source in specification["bindings"]
        if role == contract["loaderRole"]
    }
    expected_loader_values = {
        "pipelineClass": contract["pipelineClass"],
        "mode": contract["mode"],
    }
    for source in ("artifact", "repo"):
        target = binding_sources.get(source)
        if target is None:
            continue
        if _field_value(params.get(target[1])) not in contract["loaderRepositories"]:
            raise TaskTemplateContractError("Task-template graph loader identity is not exact.")
    for source, expected in expected_loader_values.items():
        target = binding_sources.get(source)
        if target is None:
            continue
        actual = _field_value(params.get(target[1]))
        if actual != expected:
            raise TaskTemplateContractError("Task-template graph loader identity is not exact.")

    for requirement in contract["requiredMedia"]:
        field = requirement["field"]
        allowed_roles = _MEDIA_FIELDS[field][1]
        if not any(role in nodes_by_role for role in allowed_roles):
            raise TaskTemplateContractError(f"Task-template graph is missing required media role for {field}.")

    output = contract["output"]
    output_node = nodes_by_role.get(output["role"])
    data = output_node.get("data") if isinstance(output_node, Mapping) else None
    if not isinstance(data, Mapping) or f"{data.get('module')}.{data.get('action')}" != output["nodeKey"]:
        raise TaskTemplateContractError("Task-template graph output identity is invalid.")
    output_id = output_node["id"]
    if any(edge.get("source") == output_id for edge in edges if isinstance(edge, Mapping)):
        raise TaskTemplateContractError("Task-template output must be a terminal graph node.")


def contracts_by_pair(contracts: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(contract["modelType"]), str(contract["mode"])): deepcopy(dict(contract))
        for contract in contracts
    }
