"""Task-first authoring over existing starters and resource inspection.

No node construction, downloads, runtime activation or execution. The returned
graph remains an ordinary operation starter; Run independently validates it.
"""

from copy import deepcopy

from modiff.operation_contracts import _identifier, operation_owns_model
from modiff.operation_starters import resolve_operation_starter


def starter_api_graph(starter):
    """Project declared inputs for the existing read-only memory planner."""
    nodes = {
        node["operation"]["operationId"]: {
            "module": node["module"], "action": node["action"],
            "params": {
                key: {"value": deepcopy(field.get("value", field.get("default")))}
                for key, field in node["params"].items() if field.get("display") != "output"
            },
        }
        for node in starter["nodes"]
    }
    for edge in starter["edges"]:
        nodes[edge["target"]]["params"][edge["targetHandle"]] = {
            "sourceId": edge["source"], "sourceKey": edge["sourceHandle"],
        }
    # Some auxiliary operations precede their consumer without preceding the
    # loader in presentation order. Derive execution order from real edges.
    order = []
    while len(order) < len(nodes):
        ready = [key for key in nodes if key not in order and all(
            not field.get("sourceId") or field["sourceId"] in order
            for field in nodes[key]["params"].values()
        )]
        if not ready:
            raise ValueError("The task starter contains a dependency cycle.")
        order.extend(ready)
    return {"nodes": nodes, "paths": [order]}


def unbind_starter(starter):
    """Keep a useful task shape, but require explicit model binding before Run."""
    result = deepcopy(starter)
    loader = next(node for node in result["nodes"] if operation_owns_model(node["operation"]))
    found = False
    for key, field in loader["params"].items():
        if field.get("display") == "model" or key in {"repo_id", "model_id", "model", "vae_model"}:
            value = {"source": "hub", "value": ""} if isinstance(field.get("value"), dict) else ""
            field.update(value=value, default=deepcopy(value), required=True)
            loader["values"][key] = value
            required = {"operationId": loader["operation"]["operationId"], "field": key}
            if required not in result["requiredInputs"]:
                result["requiredInputs"].append(required)
            found = True
        elif key in {"revision", "execution_profile_id", "reviewed_variant"}:
            field["value"] = ""
            loader["values"][key] = ""
    if not found:
        raise ValueError("This task does not yet declare an editable model input.")
    return result


def resolve_task_starter(modules, catalog, selection, *, installed, inspect_resources):
    """Choose only published compatible profiles; browser preferences grant no authority.

    `installed(profile)` checks the exact reviewed artifact. `inspect_resources`
    uses the existing workflow planner and never changes starter defaults.
    Callbacks make no-download contract tests independent of machine state.
    """
    if not isinstance(selection, dict) or not {"task"} <= set(selection) <= {
        "task", "preferredProfileId", "resourceMode",
    }:
        raise ValueError("Select a task and an optional remembered model.")
    task = _identifier(selection["task"])
    preferred = selection.get("preferredProfileId")
    if preferred is not None and (not isinstance(preferred, str) or not 0 < len(preferred) <= 256):
        raise ValueError("Invalid remembered model identity.")
    mode = selection.get("resourceMode", "auto")
    if mode not in ("auto", "expert"):
        raise ValueError("Invalid authoring memory policy.")
    profiles = {
        profile["id"]: profile for profile in catalog["diffusersExecutionProfiles"]
        if profile.get("public", True)
    }
    candidates = []
    declared = []
    for pipeline in catalog["pipelineSupport"]:
        for support in pipeline["tasks"]:
            if support["task"] != task or not support["operationIds"]:
                continue
            declared.append((pipeline["pipelineClass"], support))
            for identity in support["executionProfileIds"]:
                profile = profiles.get(identity)
                if not profile or profile["pipeline_class"] != pipeline["pipelineClass"]:
                    continue
                available = installed(profile)
                candidates.append((profile, support, available))
    if not candidates:
        if not declared:
            raise ValueError("No published model route is available for this task.")
        declared.sort(key=lambda row: (row[1]["decomposition"] != "stages", row[0]))
        starter = resolve_operation_starter(modules, catalog["operationContracts"],
                                            {"pipelineClass": declared[0][0], "task": task})
        return {"schemaVersion": 1, "starter": unbind_starter(starter), "profileId": None, "unbound": True,
                "message": "This task has an authoring contract but no published model choice. Bind a supported model before running."}
    candidates.sort(key=lambda row: (
        row[0]["id"] != preferred,
        not row[2], row[1]["dependencies"] != "ready",
        row[1]["decomposition"] != "stages", row[0]["default_repo"], row[0]["id"],
    ))
    fallback = None
    suitable = []
    for profile, support, available in candidates:
        binding = {"pipelineClass": profile["pipeline_class"], "task": task, "executionProfileId": profile["id"]}
        # Resolve even an unavailable route once for its unbound authoring shape.
        if fallback is not None and (not available or support["dependencies"] != "ready"):
            continue
        starter = resolve_operation_starter(modules, catalog["operationContracts"], binding)
        if fallback is None:
            fallback = starter
        if not available or support["dependencies"] != "ready":
            continue
        cost = float("inf")
        if mode == "auto":
            inspection = inspect_resources(starter_api_graph(starter))
            if inspection.get("canAutoRun") is not True:
                continue
            requirements = inspection.get("requirements", {})
            memory = [requirements.get(key) for key in ("systemRamBytes", "vramBytes")]
            if all(isinstance(value, (int, float)) and value >= 0 for value in memory):
                cost = sum(memory)
        suitable.append((support["decomposition"] != "stages", cost, profile["id"], starter))
        if profile["id"] == preferred:
            suitable = [suitable[-1]]
            break
    if suitable:
        _, _, identity, starter = min(suitable, key=lambda row: row[:3])
        return {"schemaVersion": 1, "starter": starter, "profileId": identity, "unbound": False,
                "message": "Using a compatible installed model. Run rechecks inputs and memory."}
    return {"schemaVersion": 1, "starter": unbind_starter(fallback), "profileId": None, "unbound": True,
            "message": "Choose a model on the loader. No installed route with a ready runtime and suitable memory recipe was selected."}
