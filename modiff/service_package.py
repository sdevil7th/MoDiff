"""Portable interface around an existing API graph; never an executor or installer."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import sys


def graph_material(graph):
    from modiff.workflow_auto_resource import graph_material as validate

    return validate(graph)


def workflow_graph_hash(graph):
    from modiff.workflow_auto_resource import workflow_graph_hash as identity

    return identity(graph)


SCHEMA = "modiff-service-v1"
LIMIT = 8 * 1024 * 1024
NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
SHA = re.compile(r"[a-f0-9]{40}\Z")
SECRET_KEY = re.compile(
    r"(?:^token$|password|secret|api[_-]?key|access[_-]?token|hf[_-]?token|authorization|credential)", re.I
)
AUTHORITY = re.compile(
    r"revision|repo|model|pipeline|class|identity|code|token|secret|password|adapter|lora|reviewed_variant|execution_profile|workflow_id|block_path|subfolder|^variant$",
    re.I,
)
MODEL_KEY = re.compile(r"(?:repo_id|model_id|model_name|model_path|pretrained_model_name_or_path)\Z")
PRIVATE_TEXT = re.compile(
    r"(?:^|\s)(?:/(?:home|Users|tmp|mnt|opt|var|etc|usr)/|[A-Za-z]:[\\/]|\\\\|~/)|hf_[A-Za-z0-9]{20,}|(?:https?://)[^\s/]+:[^\s/]+@",
    re.I,
)
SCALARS = {"str", "string", "int", "integer", "float", "number", "bool", "boolean"}
MAX_SERVICE_FILES = 128
PREVIEWS = {"ui_text", "ui_image", "ui_video", "ui_audio"}
# These are observations/correlation only. Preserve execution hints, including
# Studio specification receipts. Auto's graph-bound receipt is issued afresh.
TRANSIENT_HINTS = {
    "clientRunId",
    "runInputHash",
    "workflowTabId",
    "workflowCanvasEpoch",
    "workflowFormEpoch",
    "workflowTitle",
    "workflowSnapshot",
    "nodeId",
    "workflowAutoPlan",
}


def canonical(value):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError("Service data must be finite JSON.") from error
    if len(raw) > LIMIT:
        raise ValueError("Service package exceeds 8 MiB.")
    return raw


def digest(value):
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def portable(value, location="package", depth=0):
    if depth > 64:
        raise ValueError("Service data exceeds 64 nested levels.")
    if isinstance(value, dict):
        if value.get("source") == "local":
            raise ValueError(f"{location}: select a pinned Hub model; local model selections are not portable.")
        for key, item in value.items():
            if SECRET_KEY.search(key) and item not in (None, "", False):
                raise ValueError(f"{location}.{key}: credentials cannot be exported.")
            portable(item, f"{location}.{key}", depth + 1)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            portable(item, f"{location}[{index}]", depth + 1)
    elif isinstance(value, str):
        if PRIVATE_TEXT.search(value) or value.startswith(("/", "\\", "file:", "@data/", "@work/", "../", "./")):
            raise ValueError(f"{location}: remove the credential/local path or expose a required service input.")


def node_fields(registry, node):
    definition = registry.get(node["module"], {}).get(node["action"])
    if not isinstance(definition, dict) or not isinstance(definition.get("params"), dict):
        raise ValueError("A service node is unavailable in the current registry. Enable/review its source first.")
    return definition["params"]


def _is_preview_observation(spec, param):
    display = spec.get("display")
    return (
        isinstance(display, str)
        and display in PREVIEWS
        and bool(spec.get("dataSource"))
        and isinstance(param, dict)
        and param.get("display") == display
        and param.get("sourceKey") == spec["dataSource"]
        and not param.get("sourceId")
    )


def _modular_service_fields(graph):
    """Resolve dynamic stage fields from one concrete reviewed model owner.

    Imported field types and authoring labels are not schema authority. Reuse
    the resource-owner traversal and the same metadata used by dynamic nodes;
    ambiguous, unbound and contract-only custom paths remain undiscoverable.
    """
    nodes = graph["nodes"]
    loaders = {
        key for key, node in nodes.items()
        if (node["module"], node["action"]) == ("modules.ModularDiffusers", "ModelsLoader")
    }
    if not loaders:
        return {}
    from modiff.operation_contracts import MODULAR_STAGE_OPERATIONS
    from modiff.workflow_task_identity import resource_consumers
    from modules.ModularDiffusers.modular_utils import get_model_type_metadata

    owners = {}
    for loader in loaders:
        for consumer in resource_consumers(nodes, loader, loaders):
            owners.setdefault(consumer, []).append(loader)
    actions = {operation.action: kind for kind, operation in MODULAR_STAGE_OPERATIONS.items()}
    metadata = {}
    result = {}
    for node_id, node in nodes.items():
        if (node["module"] != "modules.ModularDiffusers" or node["action"] not in actions
                or len(owners.get(node_id, [])) != 1):
            continue
        owner = owners[node_id][0]
        if owner not in metadata:
            selector = nodes[owner]["params"].get("model_type", {})
            value = selector.get("value")
            metadata[owner] = (
                get_model_type_metadata(value)
                if isinstance(value, str) and not selector.get("sourceId") else None
            )
        definition = metadata[owner]
        if not definition or definition.get("execution_status") == "contract_only":
            continue
        stage = definition["node_params"].get(actions[node["action"]])
        if stage:
            result[node_id] = stage["params"]
    return result


def inspect_graph(graph, registry):
    graph_material(graph)
    inputs, outputs = [], []
    modular_fields = _modular_service_fields(graph)
    for node_id, node in graph["nodes"].items():
        fields = {**node_fields(registry, node), **modular_fields.get(node_id, {})}
        for field, spec in fields.items():
            if not isinstance(spec, dict):
                continue
            display = spec.get("display")
            types = spec.get("type")
            # Textarea prompts use the registry's "text" alias in ordinary
            # pipelines and "string" in Modular pipelines. Export a string
            # contract for both; this does not admit list/opaque inputs.
            if types == "text":
                types = "string"
            # File widgets may carry a scalar or a list despite a legacy
            # registry type of str. Only the trusted widget declaration can
            # grant this contract; imported parameter metadata cannot.
            if (
                isinstance(types, str) and types in {"str", "string"}
                and display == "filebrowser"
                and isinstance(spec.get("fieldOptions"), dict)
                and spec["fieldOptions"].get("multiple") is True
            ):
                types = "files"
            param = node["params"].get(field)
            if (
                _is_preview_observation(spec, param)
                and not spec.get("hidden")
                and not (node["module"] == "modules.Audio" and node["action"] == "Load")
            ):
                outputs.append({"nodeId": node_id, "field": field, "type": display})
            if (
                isinstance(types, str)
                and types in SCALARS | {"files"}
                and isinstance(param, dict)
                and not param.get("sourceId")
                and "value" in param
                and display not in (*PREVIEWS, "output", "button")
                and not AUTHORITY.search(field)
            ):
                inputs.append({"nodeId": node_id, "field": field, "type": types})
    return {"inputs": inputs, "outputs": outputs}


def bindings(interface, candidates):
    if not isinstance(interface, dict) or set(interface) != {"inputs", "outputs"}:
        raise ValueError("Interface needs named inputs and outputs maps.")
    for kind in ("inputs", "outputs"):
        selected = interface[kind]
        if not isinstance(selected, dict) or len(selected) > 128 or (kind == "outputs" and not selected):
            raise ValueError("Use at most 128 inputs/outputs and at least one persisted preview output.")
        allowed = {(x["nodeId"], x["field"]): x for x in candidates[kind]}
        seen = set()
        for name, targets in selected.items():
            if not NAME.fullmatch(name) or not isinstance(targets, list) or not targets or len(targets) > 128:
                raise ValueError(
                    "Names must start with a letter and contain letters, digits or underscores; targets must be a list."
                )
            types = set()
            for target in targets:
                if not isinstance(target, dict) or set(target) != {"nodeId", "field"}:
                    raise ValueError("A binding target needs exactly nodeId and field.")
                if not all(isinstance(v, str) for v in target.values()):
                    raise ValueError("Binding identities must be strings.")
                pair = (target["nodeId"], target["field"])
                if pair not in allowed or pair in seen:
                    raise ValueError(
                        "A binding is duplicated, connected, unavailable or not a supported service value."
                    )
                seen.add(pair)
                types.add(allowed[pair]["type"])
            if len(types) != 1:
                raise ValueError("Shared input/output targets must have the same type.")
    return deepcopy(interface)


def model_pins(graph, registry):
    from modiff.model_artifact_catalog import catalog_repository_pin

    pins = []

    def walk(value, field, location, revision=None):
        if isinstance(value, dict):
            if value.get("source") == "hub":
                walk(value.get("value"), "repo_id", location, value.get("revision") or revision)
            else:
                for key, item in value.items():
                    walk(item, key, f"{location}.{key}", value.get("revision"))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, field, f"{location}[{index}]", revision)
        elif isinstance(value, str) and value and MODEL_KEY.fullmatch(field):
            if not re.fullmatch(r"[\w.-]+/[\w.-]+", value):
                raise ValueError(f"{location}: only explicit Hub repository identifiers can be packaged.")
            pin = catalog_repository_pin(value)
            resolved = revision or (pin or {}).get("revision")
            if not isinstance(resolved, str) or not SHA.fullmatch(resolved):
                raise ValueError(f"{location}: select an immutable 40-character model revision before exporting.")
            pins.append({"location": location, "repository": value, "revision": resolved})

    for node_id, node in graph["nodes"].items():
        fields = node_fields(registry, node)
        revision = node["params"].get("revision", {}).get("value")
        for field, param in node["params"].items():
            if param.get("sourceId") and (
                MODEL_KEY.fullmatch(field) or fields.get(field, {}).get("display") == "modelselect"
            ):
                raise ValueError(f"{node_id}.{field}: service export requires a literal pinned model selection.")
            if not param.get("sourceId"):
                display = fields.get(field, {}).get("display")
                key = "repo_id" if display in ("model", "hf_model", "model_select", "modelselect") else field
                walk(param.get("value"), key, f"{node_id}.{field}", revision if MODEL_KEY.fullmatch(key) else None)
    return pins


def runtime_contract(graph, source_identity, extension_store):
    from modiff.runtime_profile import read_state, lock_digest, load_manifest, PROJECT_ROOT
    from modiff.optional_runtime_execution import graph_optional_runtime_requirement

    state = read_state(Path(sys.prefix)) or {}
    profile = state.get("profile")
    spec = load_manifest()["profiles"].get(profile)
    if not spec:
        raise ValueError("Service export requires a managed runtime profile. Use modiff.dev check.")
    contract_hash = lock_digest(PROJECT_ROOT / spec["requirements"], profile=profile)
    if state.get("lock_digest") != contract_hash:
        raise ValueError("Managed dependency contract changed; check/repair the environment before exporting.")
    custom = []
    for module in sorted({n["module"] for n in graph["nodes"].values() if n["module"].startswith("custom.")}):
        item = extension_store.require_enabled(module.removeprefix("custom."))
        custom.append({key: item[key] for key in ("moduleKey", "codeHash", "revision", "dependencies")})
    packages = {}
    for distribution in metadata.distributions():
        name = re.sub(r"[-_.]+", "-", distribution.metadata.get("Name", "")).lower()
        if name and name != "modiff":
            packages.setdefault(name, distribution.version)
    requirement = graph_optional_runtime_requirement(graph)
    return {
        "backend": {key: source_identity[key] for key in ("gitCommit", "fingerprint")},
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "profile": profile,
        "contractHash": contract_hash,
        "packages": dict(sorted(packages.items())),
        "optionalProfiles": requirement.get("profileIds", []),
        "customNodes": custom,
    }


def build_package(graph, interface, *, registry, contract):
    canonical(graph)
    candidates = inspect_graph(graph, registry)
    interface = bindings(interface, candidates)
    material = deepcopy(graph_material(graph))
    # Node parameter execution data is retained, including display/spawn flags.
    # Completed previews are observations, not inputs to the next execution.
    # Verify their registry binding rather than trusting a client display flag.
    for node in material["nodes"].values():
        for field, spec in node_fields(registry, node).items():
            param = node["params"].get(field)
            if isinstance(spec, dict) and _is_preview_observation(spec, param):
                param.pop("value", None)
    hints = graph.get("runtimeHints") or {}
    if not isinstance(hints, dict):
        raise ValueError("runtimeHints must be an object.")
    material["runtimeHints"] = {k: deepcopy(v) for k, v in hints.items() if k not in TRANSIENT_HINTS}
    if "deterministicMode" in graph:
        material["deterministicMode"] = deepcopy(graph["deterministicMode"])
    models = model_pins(material, registry)
    for targets in interface["inputs"].values():
        for target in targets:
            # Required values are supplied at invocation, never portable defaults.
            material["nodes"][target["nodeId"]]["params"][target["field"]]["value"] = None
    package = {
        "schema": SCHEMA,
        "graph": material,
        "interface": interface,
        "requirements": {**deepcopy(contract), "models": models},
    }
    portable(package)
    package["contentHash"] = digest(package)
    return package


def prepare_package(package, values, *, registry, contract, sid):
    canonical(package)
    if (
        not isinstance(package, dict)
        or set(package) != {"schema", "graph", "interface", "requirements", "contentHash"}
        or package.get("schema") != SCHEMA
    ):
        raise ValueError("Unsupported service package.")
    unsigned = {key: value for key, value in package.items() if key != "contentHash"}
    if digest(unsigned) != package["contentHash"]:
        raise ValueError("Service package content hash does not match. Export again.")
    # Rebuild validation from the real registry. A hash is integrity, not approval.
    rebuilt = build_package(package["graph"], package["interface"], registry=registry, contract=contract)
    if rebuilt != package:
        raise ValueError(
            "Service requirements changed. Match backend, packages, model pins and approved custom source, then export again."
        )
    interface = package["interface"]["inputs"]
    if not isinstance(values, dict) or set(values) != set(interface):
        raise ValueError("Supply exactly the declared named service inputs.")
    graph = deepcopy(package["graph"])
    candidates = {(x["nodeId"], x["field"]): x for x in inspect_graph(graph, registry)["inputs"]}
    for name, targets in interface.items():
        value = values[name]
        for target in targets:
            kind = candidates[(target["nodeId"], target["field"])]["type"]
            if kind == "files":
                valid = (isinstance(value, str) and bool(value)) or (
                    isinstance(value, list) and 0 < len(value) <= MAX_SERVICE_FILES
                    and all(isinstance(item, str) and bool(item) for item in value)
                )
            else:
                valid = (
                    isinstance(value, str)
                    if kind in {"str", "string"}
                    else type(value) is bool
                    if kind in {"bool", "boolean"}
                    else type(value) is int
                    if kind in {"int", "integer"}
                    else type(value) in {int, float}
                )
            if not valid:
                raise ValueError(f"Input {name} requires {kind}.")
            graph["nodes"][target["nodeId"]]["params"][target["field"]]["value"] = deepcopy(value)
    if not isinstance(sid, str) or not NAME.fullmatch(sid):
        raise ValueError("Service session ID must be a bounded identifier.")
    graph["sid"] = sid
    if graph["runtimeHints"].get("resourceMode") == "auto":
        graph["runtimeHints"]["workflowAutoPlan"] = {"schemaVersion": 1, "graphHash": workflow_graph_hash(graph)}
    graph["servicePackage"] = {"package": deepcopy(package), "values": deepcopy(values)}
    canonical(graph)
    return graph


def service_outputs(package, receipt, task_id):
    if not isinstance(receipt, dict) or not isinstance(receipt.get("outputs"), list):
        raise ValueError("Invalid run receipt.")
    result = {}
    for name, targets in package["interface"]["outputs"].items():
        selected = []
        for target in targets:
            matches = [
                item
                for item in receipt["outputs"]
                if isinstance(item, dict)
                and item.get("taskId") == task_id
                and item.get("nodeId") == target["nodeId"]
                and item.get("fieldKey") == target["field"]
            ]
            if not matches:
                raise ValueError(f"Run {task_id} did not persist output {name}. Add a preview node and run again.")
            for item in matches:
                projected = {key: item[key] for key in ("value", "url", "displayType") if key in item}
                media = item.get("mediaItems")
                if isinstance(media, list):
                    projected["mediaItems"] = [
                        {
                            key: member[key]
                            for key in ("index", "url", "displayType", "contentType", "width", "height")
                            if key in member
                        }
                        for member in media
                        if isinstance(member, dict) and member.get("taskId") == task_id
                    ]
                selected.append(projected)
        result[name] = selected
    return {"taskId": task_id, "outputs": result}


def load_json(raw):
    if len(raw) > LIMIT:
        raise ValueError("Service request exceeds 8 MiB.")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key in service request.")
            result[key] = value
        return result

    def reject_constant(_value):
        raise ValueError("Non-finite JSON.")

    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Invalid service JSON.") from error
    canonical(value)
    return value
