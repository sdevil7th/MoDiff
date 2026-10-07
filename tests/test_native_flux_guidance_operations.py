"""Ordinary Flux stages publish explicit CFG without enabling it by default."""

from unittest.mock import patch

import pytest

from modules import MODULE_MAP
from modules.ModularDiffusers.operation_contracts import get_modular_task_operation_contracts
from modiff.diffusers_profiles import public_execution_profiles
from modiff.operation_catalog import build_operation_catalog
from modiff.operation_starters import resolve_operation_starter


@pytest.mark.parametrize("pipeline", ["FluxModularPipeline", "FluxKontextModularPipeline"])
def test_flux_guidance_publication_preserves_disabled_ordinary_binding(pipeline):
    with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed model")):
        contracts = get_modular_task_operation_contracts(MODULE_MAP)
    selected = [row for row in contracts if row["pipelineClass"] == pipeline and row["operationId"] == "diffusion.guidance"]
    assert selected
    for contract in selected:
        # Public binding values identify a schema variant; resolved scalar
        # defaults are checked below through the actual starter API.
        assert contract["binding"]["values"] == {"model_type": pipeline}
        assert contract["nodeType"] == "guidance"
        assert contract["decomposition"] == "bundle"
        output = next(port for port in contract["ports"] if port["name"] == "guider_out")
        assert output["roles"] == ["component"]


@pytest.mark.parametrize("pipeline,task,profile", [
    ("FluxModularPipeline", "text_to_image", "flux-dev:modular"),
    ("FluxKontextModularPipeline", "edit_image", "flux-kontext:modular"),
])
@pytest.mark.parametrize("profiled", [False, True])
def test_profiled_and_profileless_flux_starters_keep_cfg_disabled_and_real_fanout(pipeline, task, profile, profiled):
    with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed model")):
        contracts, _ = build_operation_catalog(MODULE_MAP, public_execution_profiles(), catalog_resolver=lambda: {})
        selection = {"pipelineClass": pipeline, "task": task}
        if profiled:
            selection["executionProfileId"] = profile
        starter = resolve_operation_starter(MODULE_MAP, contracts, selection)
    nodes = {node["operation"]["operationId"]: node for node in starter["nodes"]}
    guide = nodes["diffusion.guidance"]
    assert guide["values"]["guider"] == "ClassifierFreeGuidance"
    assert guide["values"]["enabled"] is False
    assert guide["values"]["guidance_scale"] == 1.0
    assert guide["params"]["guidance_scale"]["min"] == 0.0
    denoise = nodes["diffusion.denoise"]
    assert denoise["params"]["guidance_scale"].get("hidden", False) is False
    embedded = denoise["params"]["guidance_scale"]
    assert denoise["values"].get("guidance_scale", embedded.get("value", embedded.get("default"))) == (
        2.5 if pipeline == "FluxKontextModularPipeline" else 3.5
    )
    assert {edge["target"] for edge in starter["edges"] if edge["source"] == "diffusion.guidance"} == {
        "diffusion.encode_prompt", "diffusion.denoise",
    }
