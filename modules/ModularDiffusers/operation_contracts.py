"""Operation bindings projected from the existing reviewed Modular owners."""

from copy import deepcopy

from modiff.modular_action_bindings import MODULAR_ACTION_BINDINGS, MODULAR_AUXILIARY_OPERATION_BINDINGS
from modiff.modular_block_contracts import load_reviewed_modular_block_snapshot
from modiff.modular_whole_workflow_contracts import reviewed_whole_workflow_graph_adapter
from modiff.modular_task_adapters import modular_task_adapters as _task_adapters
from modiff.modular_workflow_discovery import load_reviewed_modular_workflow_snapshot
from modiff.operation_contracts import (
    WORKFLOW_STAGE_OPERATIONS,
    build_pipeline_operation_contract,
    with_operation_semantics,
)

from .modular_utils import get_model_type_metadata, get_modular_operation_contracts


def _members(definitions, group):
    result = {}
    for definition in definitions:
        for field in definition.get(group, []):
            item = {"name": field["name"], "type": field["type"]}
            previous = result.get(item["name"])
            # Several sequential steps may deliberately refine an opaque type.
            # Keep differing representations explicit instead of picking one.
            if previous and previous["type"] != item["type"]:
                item["type"] = "opaque"
            result[item["name"]] = item
    return sorted(result.values(), key=lambda item: item["name"])


def _stage_definitions(blocks, definitions, block_name):
    return [
        definitions[p["blockDefinitionId"]]
        for p in blocks["placements"]
        if p["legacyPath"] == block_name or p["legacyPath"].startswith(block_name + ".")
    ]


def _loader_members(root, blocks, definitions, pipeline_class):
    components = _members([root], "components")
    by_name = {item["name"]: item for item in components}
    denoiser = next((name for name in ("unet", "transformer") if name in by_name), None)
    text_components = _members(_stage_definitions(blocks, definitions, "text_encoder"), "components")
    metadata = get_model_type_metadata(pipeline_class) or {}

    def selected(names):
        return [by_name[name] for name in names if name in by_name]

    return {
        "pipeline_components": components,
        "text_encoders": text_components,
        "unet_out": selected([denoiser]),
        "unet": selected([denoiser]),
        "vae_out": selected(["vae"]),
        "vae": selected(["vae"]),
        "scheduler": selected(["scheduler"]),
        "image_encoder": selected(metadata.get("loader_component_outputs", [])),
        "controlnet": selected(["controlnet"]),
    }


def get_modular_task_operation_contracts(modules) -> list[dict]:
    """Read schemas only: no node/pipeline construction, installation or weights."""
    if not modules.get("modules.ModularDiffusers"):
        return []
    generic = {(c["pipelineClass"], c["nodeKey"]): c for c in get_modular_operation_contracts(modules)}
    snapshot = load_reviewed_modular_workflow_snapshot()
    blocks_snapshot = load_reviewed_modular_block_snapshot()
    definitions = {d["id"]: d for d in blocks_snapshot["blockDefinitions"]}
    workflows = {(w["pipelineClass"], w["workflowId"]): w for w in blocks_snapshot["workflows"]}
    result = []
    for pipeline in snapshot["contracts"]:
        pipeline_class = pipeline["pipelineClass"]
        for workflow in pipeline["workflows"]:
            workflow_id = workflow["id"]
            blocks = workflows[pipeline_class, workflow_id]
            root = definitions[blocks["rootBlockDefinitionId"]]
            for task, adapter in _task_adapters(pipeline_class, workflow):
                whole = reviewed_whole_workflow_graph_adapter(pipeline_class, workflow_id)
                # A partial module registry must not advertise an executable
                # workflow whose required stage action is absent.
                node_keys = [MODULAR_ACTION_BINDINGS[key][1] for key in adapter["actionSequence"]]
                if any(
                    key.rsplit(".", 1)[1] not in modules.get(key.rsplit(".", 1)[0], {})
                    or (not whole and (pipeline_class, key) not in generic)
                    for key in node_keys
                ):
                    continue
                loader = build_pipeline_operation_contract(
                    modules,
                    pipeline_class=pipeline_class,
                    task=task,
                    operation_id="diffusion.load_models",
                    node_key="modules.ModularDiffusers.ModelsLoader",
                    loader=True,
                )
                members = _loader_members(root, blocks, definitions, pipeline_class)
                if loader:
                    for port in loader["ports"]:
                        if port["direction"] == "output" or port["name"] in {"unet", "vae", "controlnet"}:
                            port["roles"] = ["component"]
                    loader = with_operation_semantics(
                        loader,
                        workflow_id=workflow_id,
                        values={
                            "model_type": pipeline_class,
                        },
                    )
                    # Only the loader's existing allowlist can select workflow
                    # pruning. Other loaders retain their current unscoped path.
                    from .loaders import REVIEWED_BUILTIN_WORKFLOWS

                    if workflow_id in REVIEWED_BUILTIN_WORKFLOWS.get(pipeline_class, ()):
                        loader["binding"]["values"]["workflow_id"] = workflow_id
                    for port in loader["ports"]:
                        if port["name"] in members:
                            port["semantics"]["members"] = members[port["name"]]
                            port["hidden"] = not bool(members[port["name"]])
                    result.append(loader)
                helper_types = set()
                if pipeline_class == "FluxKontextModularPipeline" and task == "multi_image_reference_edit":
                    helper = build_pipeline_operation_contract(
                        modules, pipeline_class=pipeline_class, task=task,
                        operation_id="diffusion.compose_references", node_key="modules.ImageOperations.StitchImages",
                        field_overrides={"image": {"required": True}},
                    )
                    if helper is None:
                        raise ValueError("Native Kontext multi-reference requires its image composition operation.")
                    helper.update(nodeType="reference_assembly", decomposition="bundle")
                    result.append(with_operation_semantics(helper, workflow_id=workflow_id,
                                                         values={"layout": "horizontal_reference"}))
                actions = adapter["actionSequence"]
                whole = reviewed_whole_workflow_graph_adapter(pipeline_class, workflow_id)
                for index, action_key in enumerate(actions):
                    role, node_key = MODULAR_ACTION_BINDINGS[action_key]
                    if whole:
                        stage = whole["upstreamBlockSequence"][index]
                        operation_id = WORKFLOW_STAGE_OPERATIONS[stage]
                        if role == "videoEncode" and stage == "vae_encoder":
                            operation_id = "diffusion.encode_video"
                        record = build_pipeline_operation_contract(
                            modules,
                            pipeline_class=pipeline_class,
                            task=task,
                            operation_id=operation_id,
                            node_key=node_key,
                        )
                        if record is None:
                            continue
                        record.update(nodeType=stage, blockName=stage, decomposition="block")
                        for port in record["ports"]:
                            if port["name"] == "pipeline_components":
                                port["roles"] = ["component"]
                        record = with_operation_semantics(
                            record,
                            workflow_id=workflow_id,
                            values={
                                "pipeline_class": pipeline_class,
                                "workflow_id": workflow_id,
                                "block_path": stage,
                            },
                        )
                        stage_definitions = _stage_definitions(blocks, definitions, stage)
                        for port in record["ports"]:
                            semantics = port["semantics"]
                            if semantics["kind"] == "state":
                                semantics["state"] = (
                                    stage
                                    if port["direction"] == "output"
                                    else (whole["upstreamBlockSequence"][index - 1] if index else None)
                                )
                                semantics["members"] = _members(
                                    stage_definitions, "outputs" if port["direction"] == "output" else "inputs"
                                )
                            elif port["name"] == "pipeline_components":
                                semantics["members"] = _members(stage_definitions, "components")
                    else:
                        record = generic.get((pipeline_class, node_key))
                        if record is None:
                            continue
                        record = deepcopy(record)
                        record["task"] = task
                        record = with_operation_semantics(record, workflow_id=workflow_id)
                        stage_definitions = _stage_definitions(
                            blocks, definitions, record["blockName"] or record["nodeType"]
                        )
                        for port in record["ports"]:
                            if "component" in port["roles"]:
                                components = _members(stage_definitions, "components")
                                exact = [item for item in components if item["name"] == port["semanticName"]]
                                if not exact:
                                    # Generic sockets (e.g. unet) can bind a
                                    # differently named upstream component. Use
                                    # the loader's declared component projection,
                                    # not every dependency of the whole block.
                                    names = {item["name"] for item in members.get(port["name"], [])}
                                    exact = [item for item in components if item["name"] in names]
                                # The node wrapper can require a component not
                                # listed on the selected upstream block (e.g. a
                                # VAE for inpaint route validation). Preserve its
                                # exact loader projection instead of claiming all
                                # denoiser dependencies for that single socket.
                                port["semantics"]["members"] = exact or members.get(port["name"]) or components
                    # Generic node schemas keep optional media sockets for other
                    # tasks. A selected workflow can require those same sockets.
                    # Publish that requirement on the operation itself as well
                    # as the connected starter, including individually inserted nodes.
                    for port in record["ports"]:
                        if (port["direction"] == "input" and port["semantics"]["kind"] == "media"
                                and port["name"] in adapter["requiredInputs"]):
                            port["required"] = True
                    result.append(record)
                    helper_types.update(t for p in record["ports"] if p["direction"] == "input" for t in p["types"])
                for type_name in sorted(helper_types & MODULAR_AUXILIARY_OPERATION_BINDINGS.keys()):
                    operation_id, node_key = MODULAR_AUXILIARY_OPERATION_BINDINGS[type_name]
                    helper = build_pipeline_operation_contract(
                        modules, pipeline_class=pipeline_class, task=task, operation_id=operation_id, node_key=node_key
                    )
                    if helper:
                        helper.update(nodeType="reference_assembly", decomposition="bundle")
                        result.append(with_operation_semantics(helper, workflow_id=workflow_id))
                if pipeline_class == "StableDiffusionXLModularPipeline" and "custom_guider" in helper_types:
                    helper = build_pipeline_operation_contract(
                        modules, pipeline_class=pipeline_class, task=task,
                        operation_id="diffusion.guidance", node_key="modules.ModularDiffusers.Guider",
                    )
                    if helper:
                        helper.update(nodeType="guidance", decomposition="bundle")
                        for port in helper["ports"]:
                            if port["name"] == "guider_out":
                                port["roles"] = ["component"]
                        result.append(with_operation_semantics(
                            helper, workflow_id=workflow_id, values={"model_type": pipeline_class},
                        ))
                    layers = build_pipeline_operation_contract(
                        modules, pipeline_class=pipeline_class, task=task,
                        operation_id="diffusion.guidance_layers", node_key="modules.ModularDiffusers.Layers",
                    )
                    if layers:
                        layers.update(nodeType="guidance_layers", decomposition="bundle")
                        result.append(with_operation_semantics(
                            layers, workflow_id=workflow_id, values={"model_type": pipeline_class},
                        ))
    identities = [(c["pipelineClass"], c["task"], c["operationId"]) for c in result]
    if len(identities) != len(set(identities)):
        raise ValueError("Ambiguous Modular operation bindings.")
    return result
