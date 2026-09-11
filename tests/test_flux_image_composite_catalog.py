"""Ordinary FLUX composites preserve actual capabilities and one Block contract."""
from copy import deepcopy
import pytest

from modiff.huggingface_diffusers_clusters import reviewed_diffusers_cluster_catalog
from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION
from modiff.studio_execution_specs import studio_execution_spec_for_pair


def definitions():
    return [d for d in reviewed_diffusers_cluster_catalog(PINNED_DIFFUSERS_REVISION)["definitions"]
            if d["pipelineClass"] in {'FluxControlNetPipeline', 'FluxControlNetImg2ImgPipeline',
                                      'FluxControlNetInpaintPipeline', 'Flux2KleinKVPipeline'}]


@pytest.mark.parametrize("definition", definitions(), ids=lambda d: d["id"])
def test_image_composite_has_exact_inputs_dependencies_and_real_leaf_nodes(definition):
    assert definition["definitionKind"] == "studio_execution_composite"
    assert definition["outputs"][0]["name"] == "images"
    assert definition["outputs"][0]["type"] == "image"
    admission, = definition["executionAdmissions"]
    spec = studio_execution_spec_for_pair(definition["pipelineClass"], admission["studioMode"])
    assert spec["executionPath"] == "direct-diffusers-image"
    assert all(not key.startswith("modules.ModularDiffusers.") for _role, key, _x, _y in spec["roles"])
    assert "not an upstream Modular hierarchy" in definition["description"]
    inputs = {i["name"]: i for i in definition["inputs"]}
    assert inputs["seed"]["default"] == 42
    assert inputs["width"]["default"] == inputs["height"]["default"] == 1024
    kv = "KV" in definition["pipelineClass"]
    assert inputs["num_inference_steps"]["default"] == (4 if kv else 28)
    if kv:
        assert not {"guidance_scale", "negative_prompt", "conditioning_scale", "strength"} & inputs.keys()
        assert not admission["modelDependencies"]
    else:
        assert inputs["control_image"]["required"]
        dep, = admission["modelDependencies"]
        assert dep["repo"] == "InstantX/FLUX.1-dev-Controlnet-Canny"
        for key in ("kind", "repo", "revision"):
            assert admission["sealedBindingValues"][key] == dep[key]
        if definition["workflowId"] != "control_image":
            assert inputs["image"]["required"]
        if definition["workflowId"] == "control_inpaint":
            assert inputs["mask_image"]["required"]
    assert admission["dynamicFieldActions"] == []
    assert not admission["publication"]["liveProof"]
    assert not admission["publication"]["autoEligible"]


def test_image_composite_catalog_is_detached_and_has_no_duplicate_identities():
    catalog = reviewed_diffusers_cluster_catalog(PINNED_DIFFUSERS_REVISION)
    assert len(definitions()) == 6
    for key in ("definitions", "blockDefinitions"):
        assert len({d["id"] for d in catalog[key]}) == len(catalog[key])
    original = deepcopy(catalog)
    catalog["definitions"][0]["inputs"].clear()
    assert reviewed_diffusers_cluster_catalog(PINNED_DIFFUSERS_REVISION) == original
