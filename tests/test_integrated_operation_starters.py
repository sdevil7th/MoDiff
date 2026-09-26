"""An action that loads and computes must not disappear from Workflows."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from modules import MODULE_MAP
from modiff.diffusers_profiles import public_execution_profiles
from modiff.integrated_operation_contracts import get_integrated_operation_contracts
from modiff.operation_catalog import build_operation_catalog
from modiff.operation_starters import resolve_operation_starter
from modiff.upscaler_contracts import real_esrgan_x2_model_selection


SELECTION = {
    "pipelineClass": "SpandrelImageUpscaleV1",
    "task": "image_upscale",
    "executionProfileId": "real-esrgan-x2-image-upscale:direct",
}


@pytest.fixture(scope="module")
def catalog():
    return build_operation_catalog(MODULE_MAP, public_execution_profiles(), catalog_resolver=lambda: {})


def test_upscale_discovery_and_single_node_starter_do_not_load_models(catalog):
    contracts, support = catalog
    selected = next(p for p in support if p["pipelineClass"] == SELECTION["pipelineClass"])
    task = next(t for t in selected["tasks"] if t["task"] == "image_upscale")
    assert task["execution"] == "adapter"
    assert task["decomposition"] == "pipeline"
    assert task["operationIds"] == ["image.upscale"]
    assert task["executionProfileIds"] == [SELECTION["executionProfileId"]]
    original = deepcopy(MODULE_MAP["modules.Spandrel"])
    with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed model node")):
        draft = resolve_operation_starter(MODULE_MAP, contracts, SELECTION)
    assert len(draft["nodes"]) == 1
    node = draft["nodes"][0]
    assert (node["module"], node["action"]) == ("modules.Spandrel", "Upscaler")
    assert node["operation"]["decomposition"] == "integrated"
    assert node["values"]["model_id"] == real_esrgan_x2_model_selection()
    assert "revision" not in node["params"] and "pipeline_class" not in node["params"]
    assert draft["edges"] == []
    assert draft["requiredInputs"] == [{"operationId": "image.upscale", "field": "image"}]
    assert draft["sharedInputs"] == [] and draft["upstreamBlocks"] == []
    assert MODULE_MAP["modules.Spandrel"] == original


@pytest.mark.parametrize("profile", ["sdxl-base:modular", "unknown", "real-esrgan-x2-video-upscale:direct"])
def test_integrated_starter_rejects_unrelated_profiles(catalog, profile):
    with pytest.raises(ValueError):
        resolve_operation_starter(MODULE_MAP, catalog[0], {**SELECTION, "executionProfileId": profile})


def test_missing_action_is_not_advertised():
    assert get_integrated_operation_contracts({}) == []


def test_new_starter_retains_old_edits_and_does_not_share_selector_objects(catalog):
    first = resolve_operation_starter(MODULE_MAP, catalog[0], SELECTION)
    first["nodes"][0]["params"]["model_id"]["value"]["revision"] = "a" * 40
    first["nodes"][0]["params"]["downscale"]["value"] = 0.5
    second = resolve_operation_starter(MODULE_MAP, catalog[0], SELECTION)
    assert second["nodes"][0]["params"]["model_id"]["value"] == real_esrgan_x2_model_selection()
    assert first["nodes"][0]["params"]["downscale"]["value"] == 0.5
