"""Selected stage and retained whole starters use only their actual route nodes."""
from copy import deepcopy
from unittest.mock import patch

import pytest

PAIRS = (
    ("FluxReduxPipeline", "edit_image"),
    ("FluxReduxPipeline", "multi_image_reference_edit"),
    ("FluxFillPipeline", "inpaint"),
    ("FluxFillPipeline", "outpaint"),
    ("FluxCannyPipeline", "control_image"),
    ("FluxDepthPipeline", "control_image"),
)
STAGE_TYPES = {"image_encoder", "denoise", "decoder"}


@pytest.fixture(scope="module")
def catalog():
    from modules import MODULE_MAP
    from modiff.operation_catalog import build_operation_catalog
    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    return MODULE_MAP, contracts


def resolve(catalog, model_type, task, *, profile=True, repository=None):
    from modiff.operation_starters import resolve_operation_starter
    from modiff.studio_execution_specs import studio_execution_spec_for_pair
    spec = studio_execution_spec_for_pair(model_type, task)
    selection = {"pipelineClass": spec["pipelineClass"], "task": task}
    if profile:
        selection["executionProfileId"] = spec["executionProfileId"]
    if repository is not None:
        selection["repository"] = repository
    with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed executable node")):
        return resolve_operation_starter(*catalog, selection)


def assert_closed(result):
    ids = {node["operation"]["operationId"] for node in result["nodes"]}
    assert len(ids) == len(result["nodes"])
    for edge in result["edges"]:
        assert edge["source"] in ids and edge["target"] in ids
    for row in result["requiredInputs"]:
        assert row["operationId"] in ids, "Excluded route inputs must not be published"
    for group in result["sharedInputs"]:
        assert all(member["operationId"] in ids for member in group["members"])


@pytest.mark.parametrize("model_type,task", PAIRS)
def test_exact_selected_image_starter_has_three_real_stages_and_retains_whole_catalog(catalog, model_type, task):
    before = deepcopy(catalog[1])
    result = resolve(catalog, model_type, task)
    assert_closed(result)
    assert len(result["nodes"]) == 4
    owner, *stages = result["nodes"]
    assert owner["operation"]["decomposition"] == "loader"
    assert {row["operation"]["nodeType"] for row in stages} == STAGE_TYPES
    assert all(row["operation"]["decomposition"] == "bundle" for row in stages)
    selected = [row for row in catalog[1] if row["pipelineClass"] == result["pipelineClass"] and row["task"] == task]
    assert len(selected) == 5
    assert len([row for row in selected if row["decomposition"] == "pipeline"]) == 1
    assert catalog[1] == before


@pytest.mark.parametrize("model_type,task", PAIRS)
def test_unbound_old_whole_starter_does_not_publish_excluded_stage_inputs(catalog, model_type, task):
    result = resolve(catalog, model_type, task, profile=False)
    assert_closed(result)
    assert len(result["nodes"]) == 2
    assert [row["operation"]["decomposition"] for row in result["nodes"]] == ["loader", "pipeline"]
    assert len(result["edges"]) == 1
    assert result["edges"][0]["sourceHandle"] == result["edges"][0]["targetHandle"] == "pipeline"


def test_reviewed_canny_repair_retains_whole_route_without_stage_grant(catalog):
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
    from modiff.model_artifact_catalog import require_catalog_revision
    profile = DIFFUSERS_EXECUTION_PROFILES["flux-canny:direct"]
    repository, = profile.compatible_repos
    result = resolve(catalog, "FluxCannyPipeline", "control_image", repository=repository)
    assert_closed(result)
    assert len(result["nodes"]) == 2
    owner, whole = result["nodes"]
    assert whole["action"] == "ControlGenerate"
    assert owner["values"]["model_id"] == {"source": "hub", "value": repository}
    assert owner["values"]["revision"] == require_catalog_revision(repository, model_type=profile.model_type)
    assert whole["operation"]["decomposition"] == "pipeline"


@pytest.mark.parametrize("corruption", ["missing_stage", "extra_pipeline"])
def test_fallback_does_not_silently_prune_an_incomplete_or_extra_partition(catalog, corruption):
    from modiff.operation_starters import resolve_operation_starter
    from modiff.studio_execution_specs import studio_execution_spec_for_pair
    spec = studio_execution_spec_for_pair("FluxFillPipeline", "inpaint")
    contracts = deepcopy(catalog[1])
    selected = [row for row in contracts if row["pipelineClass"] == spec["pipelineClass"]
                and row["task"] == "inpaint"]
    if corruption == "missing_stage":
        encoder, = [row for row in selected if row["nodeType"] == "image_encoder"]
        contracts.remove(encoder)
    else:
        whole, = [row for row in selected if row["decomposition"] == "pipeline"]
        extra = deepcopy(whole)
        extra["operationId"] = "image.extra_whole"
        contracts.append(extra)
    with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed executable node")):
        with pytest.raises(ValueError, match="alternatives are incomplete or ambiguous"):
            resolve_operation_starter(catalog[0], contracts,
                {"pipelineClass": spec["pipelineClass"], "task": "inpaint"})


@pytest.mark.parametrize("pipeline,task,expected", (
    ("AnimaModularPipeline", "text_to_image", 4),
    ("FluxModularPipeline", "text_to_image", 5),
    ("FluxModularPipeline", "image_to_image", 6),
))
def test_native_and_auxiliary_starter_nodes_are_not_pruned(catalog, pipeline, task, expected):
    from modiff.operation_starters import resolve_operation_starter
    with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed executable node")):
        result = resolve_operation_starter(*catalog, {"pipelineClass": pipeline, "task": task})
    assert_closed(result)
    assert len(result["nodes"]) == expected
    if pipeline == "FluxModularPipeline":
        assert any(row["operation"]["operationId"] == "diffusion.guidance" for row in result["nodes"])
