"""Small ordinary graph drafts derived from existing operation/workflow owners.

This is an authoring projection, not an execution recipe or qualification receipt.
Unbound conditioning/components remain explicit. No models are constructed here.
"""

from modiff.operation_catalog import resolve_operation
from modiff.operation_contracts import _identifier


def _modular_route(pipeline, task):
    from modiff.modular_workflow_discovery import load_reviewed_modular_workflow_snapshot
    from modiff.modular_task_adapters import modular_task_adapters

    for entry in load_reviewed_modular_workflow_snapshot()["contracts"]:
        if entry["pipelineClass"] == pipeline:
            for workflow in entry["workflows"]:
                for name, adapter in modular_task_adapters(pipeline, workflow):
                    if name == task:
                        return workflow["id"], adapter
    raise ValueError("No reviewed stage route for this task. Add individual operations instead.")


def _bind_execution_profile(loader, task, identity):
    """Select an existing reviewed loader profile without importing model code."""
    from copy import deepcopy
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES, resolve_execution_profiles_for_loader
    from modiff.model_artifact_catalog import require_catalog_revision

    if not isinstance(identity, str) or identity != identity.strip():
        raise ValueError("Select an exact execution profile.")
    profile = DIFFUSERS_EXECUTION_PROFILES.get(identity)
    if (
        profile is None
        or not profile.public
        or profile.pipeline_class != loader["operation"]["binding"]["pipelineClass"]
        or profile.backend_path != f"{loader['module']}.{loader['action']}"
        or (profile.execution_path != "modular-diffusers" and task not in profile.modes)
    ):
        raise ValueError("The execution profile does not belong to this pipeline/task.")
    params = loader["params"]
    repo_field = "repo_id" if loader["action"] == "ModelsLoader" else "model_id"
    if repo_field not in params or "revision" not in params:
        raise ValueError("This loader does not expose a reviewed model selection.")
    values = {
        repo_field: {"source": "hub", "value": profile.default_repo},
        "revision": require_catalog_revision(profile.default_repo, model_type=profile.model_type),
    }
    if "execution_profile_id" in params:
        values["execution_profile_id"] = profile.id
    if "reviewed_variant" in params:
        options = params["reviewed_variant"].get("options", [])
        values["reviewed_variant"] = profile.default_repo if profile.default_repo in options else ""
    for key, value in values.items():
        params[key]["value"] = deepcopy(value)
        loader["values"][key] = deepcopy(value)
    resolved, reason = resolve_execution_profiles_for_loader(loader["module"], loader["action"], loader["values"])
    if reason or profile.id not in {p.id for p in resolved}:
        raise ValueError("The selected model does not resolve to its reviewed loader profile.")


def resolve_operation_starter(modules, contracts, selection):
    if not isinstance(selection, dict) or set(selection) not in (
        {"pipelineClass", "task"},
        {"pipelineClass", "task", "executionProfileId"},
    ):
        raise ValueError("Select an exact pipeline and task.")
    pipeline, task = (_identifier(selection[key]) for key in ("pipelineClass", "task"))
    binding = {"pipelineClass": pipeline, "task": task}
    selected = [c for c in contracts if c["pipelineClass"] == pipeline and c["task"] == task]
    if not selected or sum(c["decomposition"] == "loader" for c in selected) != 1:
        raise ValueError("No complete operation binding for this pipeline/task.")
    nodes = {
        c["operationId"]: resolve_operation(modules, contracts, {**binding, "operationId": c["operationId"]})
        for c in selected
    }
    edges = []
    targets = set()

    def connect(source, source_handle, target, target_handle):
        from modiff.block_definition_v2 import _block_value_types_are_compatible_v2

        left = nodes[source]["params"].get(source_handle)
        right = nodes[target]["params"].get(target_handle)
        if (
            not left
            or not right
            or left.get("hidden")
            or right.get("hidden")
            or left.get("display") != "output"
            or not (right.get("display") == "input" or right.get("isInput"))
            or not _block_value_types_are_compatible_v2(left.get("type"), right.get("type"))
        ):
            raise ValueError(
                f"The declared connection {source}.{source_handle} → {target}.{target_handle} is unavailable."
            )
        edge = {"source": source, "sourceHandle": source_handle, "target": target, "targetHandle": target_handle}
        if edge in edges:
            return
        if (target, target_handle) in targets:
            raise ValueError("The starter has competing input connections.")
        edges.append(edge)
        targets.add((target, target_handle))

    loader = next(c for c in selected if c["decomposition"] == "loader")["operationId"]
    if "executionProfileId" in selection:
        _bind_execution_profile(nodes[loader], task, selection["executionProfileId"])
    workflow_id, upstream, required = None, [], set()
    ordered = [loader]
    if any(c["decomposition"] == "pipeline" for c in selected):
        for c in selected:
            if c["decomposition"] == "pipeline":
                connect(loader, "pipeline", c["operationId"], "pipeline")
                ordered.append(c["operationId"])
    else:
        from modiff.modular_action_bindings import MODULAR_ACTION_BINDINGS
        from modiff.huggingface_cluster_admission import _REQUIRED_COMPONENT_EDGES, _PIPELINE_ACTION_COMPONENT_EDGES

        workflow_id, adapter = _modular_route(pipeline, task)
        upstream = adapter["upstreamBlockSequence"]
        required = set(adapter["requiredInputs"])
        by_key = {c["nodeKey"]: c["operationId"] for c in selected}
        actions = {a: by_key[MODULAR_ACTION_BINDINGS[a][1]] for a in adapter["actionSequence"]}
        ordered.extend(actions.values())
        for action, operation in actions.items():
            # These exact component edges already govern registered admission.
            # Auxiliary loaders/guider nodes are deliberately left as inputs.
            component_edges = _REQUIRED_COMPONENT_EDGES.get(action, set()) | _PIPELINE_ACTION_COMPONENT_EDGES.get(
                (pipeline, action), set()
            )
            for source, output, _, input_ in sorted(component_edges):
                if source == "models":
                    connect(loader, output, operation, input_)
        for edge in adapter["stateEdges"]:
            connect(
                actions[edge["producerAction"]],
                edge["producerOutput"],
                actions[edge["consumerAction"]],
                edge["consumerInput"],
            )
        # Publish effective output dimensions instead of duplicating decoder
        # literals. The generic config declares their semantic names explicitly.
        for c in selected:
            for output in c["ports"]:
                if (
                    output["direction"] != "output"
                    or output["hidden"]
                    or output["semanticName"] not in {"width", "height"}
                ):
                    continue
                for target in selected:
                    if target["operationId"] != "diffusion.decode_latents":
                        continue
                    for input_ in target["ports"]:
                        if (
                            input_["direction"] == "input"
                            and not input_["hidden"]
                            and input_["semanticName"] == output["semanticName"]
                        ):
                            connect(c["operationId"], output["name"], target["operationId"], input_["name"])
        # Typed ordinary auxiliary operations, e.g. a reference assembler.
        for c in selected:
            if c["operationId"] in ordered:
                continue
            ordered.insert(1, c["operationId"])
            for output in c["ports"]:
                if output["direction"] != "output" or output["hidden"]:
                    continue
                matches = [
                    (target, port)
                    for target in selected
                    if target is not c
                    for port in target["ports"]
                    if port["direction"] == "input" and not port["hidden"] and port["types"] == output["types"]
                ]
                if len(matches) == 1:
                    target, port = matches[0]
                    connect(c["operationId"], output["name"], target["operationId"], port["name"])

    # Admission describes minimum reviewed paths. The operation owner also
    # declares required model inputs (e.g. SDXL Denoise's VAE). Fill only exact,
    # unique component bundles from this loader; never infer from tensor types.
    loader_ports = nodes[loader]["operation"]["ports"]
    for operation, node in nodes.items():
        if operation == loader:
            continue
        for input_ in node["operation"]["ports"]:
            members = input_["semantics"]["members"]
            if (
                input_["direction"] != "input"
                or input_["hidden"]
                or not input_["required"]
                or "component" not in input_["roles"]
                or not members
                or (operation, input_["name"]) in targets
            ):
                continue
            matches = [
                p
                for p in loader_ports
                if p["direction"] == "output"
                and not p["hidden"]
                and "component" in p["roles"]
                and p["semantics"]["members"] == members
                and p["types"] == input_["types"]
            ]
            if len(matches) == 1:
                connect(loader, matches[0]["name"], operation, input_["name"])

    # Both native PipelineState continuations and sealed route states carry one
    # generator. Expose its shared seed as an authoring relationship; execution
    # still receives ordinary values and retains the existing seed/state checks.
    state_groups = [{key} for key in nodes]
    for edge in edges:
        output = next(
            p
            for p in nodes[edge["source"]]["operation"]["ports"]
            if p["name"] == edge["sourceHandle"] and p["direction"] == "output"
        )
        if output["semantics"]["kind"] != "state":
            continue
        joined = [g for g in state_groups if edge["source"] in g or edge["target"] in g]
        state_groups = [g for g in state_groups if g not in joined]
        state_groups.append(set().union(*joined))
    shared_inputs = []
    for group in state_groups:
        members = [
            {"operationId": key, "field": p["name"]}
            for key in ordered
            if key in group
            for p in nodes[key]["operation"]["ports"]
            if p["direction"] == "input"
            and p["semanticName"] == "seed"
            and not p["hidden"]
            and p["semantics"]["kind"] == "value"
        ]
        if len(members) > 1:
            shared_inputs.append({"name": "seed", "members": members})

    required_inputs = []
    for operation, node in nodes.items():
        declared_required = {
            p["name"] for p in node["operation"]["ports"] if p["required"] and p["direction"] == "input"
        }
        for name, field in node["params"].items():
            if field.get("hidden") or field.get("display") == "output" or (operation, name) in targets:
                continue
            if field.get("required") or name in required or name in declared_required:
                required_inputs.append({"operationId": operation, "field": name})
    return {
        "schemaVersion": 1,
        **binding,
        "workflowId": workflow_id,
        "nodes": [nodes[key] for key in ordered],
        "edges": edges,
        "requiredInputs": required_inputs,
        "sharedInputs": shared_inputs,
        "upstreamBlocks": upstream,
    }
