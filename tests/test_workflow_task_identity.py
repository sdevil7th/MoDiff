from copy import deepcopy

import pytest

from modiff.workflow_auto_resource import build_workflow_auto_plan


def modular_graph(pipeline, mode):
    from modiff.modular_action_bindings import MODULAR_ACTION_BINDINGS
    from modiff.modular_workflow_contracts import PINNED_MODULAR_WORKFLOW_TRUTH

    route = PINNED_MODULAR_WORKFLOW_TRUTH[pipeline].mode(mode)
    nodes = {
        "loader": {
            "module": "modules.ModularDiffusers",
            "action": "ModelsLoader",
            "params": {
                "model_type": {"value": pipeline},
                "repo_id": {"value": "test/model"},
                "device": {"value": "cpu"},
                "dtype": {"value": "float32"},
            },
        }
    }
    for action in route.action_sequence:
        module, name = MODULAR_ACTION_BINDINGS[action][1].rsplit(".", 1)
        nodes[action] = {
            "module": module,
            "action": name,
            "params": {
                "pipeline_components": {"sourceId": "loader", "sourceKey": "pipeline_components"},
            },
        }
    for edge in route.state_edges:
        nodes[edge.consumer_action]["params"][edge.consumer_input] = {
            "sourceId": edge.producer_action,
            "sourceKey": edge.producer_output,
        }
    # Only the task's actual external input set is supplied.
    for name in route.required_upstream_inputs:
        nodes[route.action_sequence[0]]["params"][name] = {"value": "test-input"}
    return {"nodes": nodes, "paths": [list(nodes)]}


def test_auto_resolves_sdxl_img2img_from_the_executable_graph(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from modiff import workflow_auto_resource as planner

    profile = SimpleNamespace(
        model_type="StableDiffusionXLModularPipeline",
        modes=("text_to_image", "image_to_image"),
        loader_module="modules.ModularDiffusers",
        loader_action="ModelsLoader",
        execution_path="modular-diffusers",
    )
    monkeypatch.setattr(
        planner,
        "resolve_execution_profiles_for_loader",
        lambda module, action, values: ((profile,), None) if action == "ModelsLoader" else ((), None),
    )
    requests = []

    def recipe(payload, **kwargs):
        requests.append(payload)
        return {"candidates": []}

    graph = modular_graph(profile.model_type, "image_to_image")
    before = deepcopy(graph)
    result = build_workflow_auto_plan(
        graph,
        runtime_fingerprint={},
        local_models=[],
        data_dir=str(tmp_path),
        plan_recipe=recipe,
        hardware={"systemMemory": {}, "accelerator": {}, "offloadDisk": {}},
    )
    assert requests and requests[0]["form"]["mode"] == "image_to_image"
    assert not result["canAutoRun"], "Resolving a task does not grant an unqualified resource recipe."
    assert graph == before


@pytest.mark.parametrize(
    "pipeline,task",
    [
        ("StableDiffusionXLModularPipeline", "text_to_image"),
        ("StableDiffusionXLModularPipeline", "image_to_image"),
        ("StableDiffusionXLModularPipeline", "inpaint"),
        ("FluxModularPipeline", "text_to_image"),
        ("FluxModularPipeline", "image_to_image"),
        ("QwenImageModularPipeline", "text_to_image"),
    ],
)
def test_reviewed_tasks_are_identified_without_authoring_hints(pipeline, task):
    from modiff.workflow_task_identity import modular_graph_tasks

    graph = modular_graph(pipeline, task)
    assert modular_graph_tasks(graph["nodes"], "loader", list(graph["nodes"])[1:], pipeline) == {task}


def test_missing_or_wrong_state_wire_and_duplicate_consumers_do_not_guess():
    from modiff.workflow_task_identity import modular_graph_tasks

    pipeline = "StableDiffusionXLModularPipeline"
    graph = modular_graph(pipeline, "image_to_image")["nodes"]

    def tasks(nodes):
        return modular_graph_tasks(nodes, "loader", list(nodes)[1:], pipeline)

    assert tasks(graph) == {"image_to_image"}
    broken = deepcopy(graph)
    edge = next(p for p in broken["denoise"]["params"].values() if p.get("sourceId") == "vae_encoder")
    edge["sourceKey"] = "unrelated"
    assert tasks(broken) == set()
    duplicate = deepcopy(graph)
    duplicate["other_denoise"] = deepcopy(graph["denoise"])
    assert tasks(duplicate) == set()
    assert modular_graph_tasks(graph, "loader", ["denoise"], pipeline) == set()
    assert modular_graph_tasks(graph, "loader", list(graph)[1:], "UnknownPipeline") == set()


def test_actual_operation_starter_edges_resolve_the_selected_image_task():
    from modules import MODULE_MAP
    from modiff.operation_catalog import build_operation_catalog
    from modiff.operation_starters import resolve_operation_starter
    from modiff.workflow_task_identity import modular_graph_tasks

    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    for pipeline, task in [
        ("StableDiffusionXLModularPipeline", "text_to_image"),
        ("StableDiffusionXLModularPipeline", "image_to_image"),
        ("FluxModularPipeline", "text_to_image"),
        ("FluxModularPipeline", "image_to_image"),
        ("QwenImageModularPipeline", "text_to_image"),
    ]:
        starter = resolve_operation_starter(MODULE_MAP, contracts, {"pipelineClass": pipeline, "task": task})
        nodes = {
            n["operation"]["operationId"]: {
                "module": n["module"],
                "action": n["action"],
                "params": {
                    k: {"value": p.get("value", p.get("default"))}
                    for k, p in n["params"].items()
                    if p.get("display") != "output"
                },
            }
            for n in starter["nodes"]
        }
        for e in starter["edges"]:
            nodes[e["target"]]["params"][e["targetHandle"]] = {"sourceId": e["source"], "sourceKey": e["sourceHandle"]}
        loader = starter["nodes"][0]["operation"]["operationId"]
        assert modular_graph_tasks(nodes, loader, list(nodes)[1:], pipeline) == {task}, (pipeline, task)


def test_output_history_uses_recognized_graph_task_without_rewriting_form():
    from modiff.execution_input_provenance import (
        capture_generation_inputs,
        build_resolved_execution_inputs,
        apply_resolved_execution_inputs,
    )

    graph = modular_graph("StableDiffusionXLModularPipeline", "image_to_image")
    records = {
        key: capture_generation_inputs(
            key, node, {field: p.get("value") for field, p in node["params"].items() if "value" in p}
        )
        for key, node in graph["nodes"].items()
    }
    receipt = build_resolved_execution_inputs(graph, records, task_id="run", attempt_index=0, node_id="decoder")
    output = {"taskId": "run", "nodeId": "decoder", "mode": "text_to_image", "formSnapshot": {"mode": "text_to_image"}}
    result = apply_resolved_execution_inputs(output, receipt)
    assert result["mode"] == "image_to_image"
    assert result["formSnapshot"]["mode"] == "text_to_image"
    assert receipt["graphTasks"] == [
        {"loaderId": "loader", "pipelineClass": "StableDiffusionXLModularPipeline", "task": "image_to_image"}
    ]


def test_conflicting_explicit_modes_remain_unresolved_even_for_a_single_mode_profile(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from modiff import workflow_auto_resource as planner

    profile = SimpleNamespace(
        model_type="StableDiffusionXLModularPipeline",
        modes=("text_to_image",),
        loader_module="modules.ModularDiffusers",
        loader_action="ModelsLoader",
        execution_path="modular-diffusers",
    )
    monkeypatch.setattr(
        planner,
        "resolve_execution_profiles_for_loader",
        lambda module, action, values: ((profile,), None) if action == "ModelsLoader" else ((), None),
    )
    graph = modular_graph(profile.model_type, "text_to_image")
    graph["nodes"]["loader"]["params"]["mode"] = {"value": "text_to_image"}
    graph["nodes"]["denoise"]["params"]["mode"] = {"value": "image_to_image"}

    def recipe(*args, **kwargs):
        raise AssertionError("Conflicting tasks must not reach recipe selection")

    result = planner.build_workflow_auto_plan(
        graph,
        runtime_fingerprint={},
        local_models=[],
        data_dir=str(tmp_path),
        plan_recipe=recipe,
        hardware={"systemMemory": {}, "accelerator": {}, "offloadDisk": {}},
    )
    assert not result["canAutoRun"] and any("ambiguous" in issue for issue in result["issues"])


def test_ambiguous_declared_tasks_missing_owner_capture_and_unrelated_nodes_stay_explicit(monkeypatch):
    from dataclasses import replace
    from modiff import workflow_task_identity as identity

    pipeline = "StableDiffusionXLModularPipeline"
    nodes = modular_graph(pipeline, "text_to_image")["nodes"]
    truth = identity.PINNED_MODULAR_WORKFLOW_TRUTH[pipeline]
    monkeypatch.setitem(
        identity.PINNED_MODULAR_WORKFLOW_TRUTH,
        pipeline,
        replace(truth, modes=(*truth.modes, ("other_task", truth.mode("text_to_image")))),
    )
    assert identity.modular_graph_tasks(nodes, "loader", list(nodes)[1:], pipeline) == {"text_to_image", "other_task"}
    records = {"loader": {"fields": {"model_type": {"value": pipeline}}}}
    assert identity.graph_task_receipts(nodes, records)[0]["task"] is None
    assert identity.graph_task_receipts(nodes, {}) == []
    nodes["unrelated"] = {"module": "custom.Developer", "action": "Process", "params": {}}
    assert identity.resource_consumers(nodes, "loader", {"loader"}) == ["decoder", "denoise", "text_encoder"]


def test_receipts_use_only_the_output_ancestry_and_leave_incomplete_graphs_unlabeled():
    from modiff.execution_input_provenance import capture_generation_inputs, build_resolved_execution_inputs

    graph = modular_graph("StableDiffusionXLModularPipeline", "image_to_image")
    graph["nodes"]["other_loader"] = deepcopy(graph["nodes"]["loader"])
    records = {
        key: capture_generation_inputs(
            key, node, {field: p.get("value") for field, p in node["params"].items() if "value" in p}
        )
        for key, node in graph["nodes"].items()
    }
    receipt = build_resolved_execution_inputs(graph, records, task_id="run", attempt_index=0, node_id="decoder")
    assert [row["loaderId"] for row in receipt["graphTasks"]] == ["loader"]
    del records["vae_encoder"]
    receipt = build_resolved_execution_inputs(graph, records, task_id="run", attempt_index=0, node_id="decoder")
    assert "graphTasks" not in receipt


@pytest.mark.parametrize(
    "pipeline,repository,task,resource_mode",
    [
        (
            "StableDiffusionXLModularPipeline",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "image_to_image",
            "edit_image",
        ),
        (
            "StableDiffusionXLModularPipeline",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "text_to_image",
            "text_to_image",
        ),
        ("FluxModularPipeline", "black-forest-labs/FLUX.1-dev", "image_to_image", "image_to_image"),
        ("QwenImageModularPipeline", "Qwen/Qwen-Image-2512", "text_to_image", "modular_text_to_image"),
    ],
)
def test_actual_profiles_use_their_existing_resource_mode_alias(tmp_path, pipeline, repository, task, resource_mode):
    graph = modular_graph(pipeline, task)
    graph["nodes"]["loader"]["params"]["repo_id"] = {"value": {"source": "hub", "value": repository}}
    requests = []

    def recipe(payload, **kwargs):
        requests.append(payload)
        return {"candidates": []}

    result = build_workflow_auto_plan(
        graph,
        runtime_fingerprint={},
        local_models=[],
        data_dir=str(tmp_path),
        plan_recipe=recipe,
        hardware={"systemMemory": {}, "accelerator": {}, "offloadDisk": {}},
    )
    assert requests and requests[0]["form"]["mode"] == resource_mode
    assert not result["canAutoRun"], "The existing resource profile still needs a valid recipe."
