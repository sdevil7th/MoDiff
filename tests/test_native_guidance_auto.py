"""Native CFG suppliers remain in the exact model owner's Auto scope."""

from copy import deepcopy
from unittest.mock import patch

import pytest

from modiff.diffusers_profiles import resolve_execution_profiles_for_loader
from modiff.operation_catalog import build_operation_catalog
from modiff.operation_starters import resolve_operation_starter
from modiff.workflow_auto_resource import build_workflow_auto_plan
from modules import MODULE_MAP


@pytest.fixture(scope="module")
def contracts():
    return build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})[0]


def starter_graph(contracts, pipeline, profile_id):
    starter = resolve_operation_starter(MODULE_MAP, contracts, {
        "pipelineClass": pipeline, "task": "text_to_image", "executionProfileId": profile_id,
    })
    nodes = {
        node["operation"]["operationId"]: {
            "module": node["module"], "action": node["action"],
            "params": {
                key: {"value": param.get("value", param.get("default"))}
                for key, param in node["params"].items() if param.get("display") != "output"
            },
        }
        for node in starter["nodes"]
    }
    for edge in starter["edges"]:
        nodes[edge["target"]]["params"][edge["targetHandle"]] = {
            "sourceId": edge["source"], "sourceKey": edge["sourceHandle"],
        }
    order = []
    while len(order) < len(nodes):
        ready = [key for key, node in nodes.items() if key not in order and {
            param["sourceId"] for param in node["params"].values() if param.get("sourceId")
        }.issubset(order)]
        assert ready, "The actual starter must have an executable dependency order."
        order.extend(ready)
    return {"nodes": nodes, "paths": [order]}


def plan_graph(graph, tmp_path):
    loader = next(node for node in graph["nodes"].values() if node["action"] == "ModelsLoader")
    values = {key: param.get("value") for key, param in loader["params"].items() if not param.get("sourceId")}
    profiles, reason = resolve_execution_profiles_for_loader(loader["module"], loader["action"], values)
    assert reason is None and len(profiles) == 1
    profile = profiles[0]

    def recipe(payload, **kwargs):
        # Inject only a bounded resource fixture. This test establishes graph
        # ownership, never model availability, measured fit or GPU qualification.
        form = payload["form"]
        assert form["executionProfileId"] == profile.id
        return {"candidates": [{
            "id": "guidance-ownership-fixture", "modelRepo": form["modelRepo"],
            "executionProfileId": profile.id,
            "modelType": form["modelType"], "mode": form["mode"],
            "loaderModule": profile.loader_module, "loaderAction": profile.loader_action,
            "executionPath": profile.execution_path, "dtype": form["dtype"],
            "quantizationMode": form["quantizationMode"], "offloadMode": form["offloadMode"],
            "autoOffload": form["autoOffload"], "canAutoRun": True,
            "proof": {"status": "declared_safe"}, "artifactRevision": values["revision"],
            "generation": {key: form[key] for key in (
                "width", "height", "steps", "guidanceScale", "batchSize", "maxSequenceLength",
            ) if key in form},
            "requirements": {"systemRamBytes": 300, "vramBytes": 200, "diskFreeBytes": 0},
        }]}

    return build_workflow_auto_plan(
        graph, runtime_fingerprint={}, local_models=[], data_dir=str(tmp_path), plan_recipe=recipe,
        hardware={"accelerator": {"freeBytes": 1000, "memoryKind": "dedicated"},
                  "systemMemory": {"availableBytes": 1000}, "offloadDisk": {"freeBytes": 1000}},
    )


@pytest.mark.parametrize("pipeline,profile_id", [
    ("QwenImageModularPipeline", "qwen-image:modular"),
    ("ZImageModularPipeline", "z-image:modular"),
    ("FluxModularPipeline", "flux-dev:modular"),
    ("FluxKontextModularPipeline", "flux-kontext:modular"),
])
def test_actual_cfg_starter_is_owned_without_node_construction(contracts, tmp_path, pipeline, profile_id):
    with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed during planning")):
        graph = starter_graph(contracts, pipeline, profile_id)
        before = deepcopy(graph)
        result = plan_graph(graph, tmp_path)
    assert result["canAutoRun"], result["issues"]
    assert "diffusion.guidance" in result["loaders"][0]["consumers"]
    for target in ("diffusion.encode_prompt", "diffusion.denoise"):
        assert graph["nodes"][target]["params"]["guider"] == {
            "sourceId": "diffusion.guidance", "sourceKey": "guider_out",
        }
    assert graph == before


@pytest.mark.parametrize("mutation", ["disconnected", "wrong_output", "unknown_supplier"])
def test_cfg_ownership_requires_the_exact_reviewed_connections(contracts, tmp_path, mutation):
    graph = starter_graph(contracts, "QwenImageModularPipeline", "qwen-image:modular")
    if mutation == "unknown_supplier":
        graph["nodes"]["diffusion.guidance"]["module"] = "modules.UnknownGuidance"
    else:
        for target in ("diffusion.encode_prompt", "diffusion.denoise"):
            param = graph["nodes"][target]["params"]["guider"]
            if mutation == "disconnected":
                param.clear()
            else:
                param["sourceKey"] = "unreviewed_output"
    result = plan_graph(graph, tmp_path)
    assert not result["canAutoRun"]
    assert any("diffusion.guidance" in issue for issue in result["issues"])
