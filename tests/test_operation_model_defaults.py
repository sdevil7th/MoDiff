"""New generic workflows must retain the selected model's reviewed defaults."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from modules import MODULE_MAP
from modiff.operation_catalog import build_operation_catalog, resolve_operation
from modiff.operation_starters import resolve_operation_starter


@pytest.fixture(scope="module")
def contracts():
    return build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})[0]


@pytest.mark.parametrize("profile,pipeline,steps,guidance,size,dtype", [
    ("lcm-dreamshaper-v7:direct", "LatentConsistencyModelPipeline", 4, 8.5, 512, "float32"),
    ("flux-schnell:direct", "FluxPipeline", 4, 0.0, 1024, "bfloat16"),
    ("flux-krea:direct", "FluxPipeline", 28, 3.5, 1024, "bfloat16"),
    ("pixart-sigma-1024:direct", "PixArtSigmaPipeline", 20, 4.5, 1024, "float32"),
])
def test_profile_starter_uses_reviewed_values_without_constructing_models(
    contracts, profile, pipeline, steps, guidance, size, dtype,
):
    original = deepcopy(MODULE_MAP["modules.DiffusersImage"])
    with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed model node")):
        starter = resolve_operation_starter(MODULE_MAP, contracts, {
            "pipelineClass": pipeline, "task": "text_to_image", "executionProfileId": profile,
        })
    loader = next(n for n in starter["nodes"] if n["action"] == "LoadPipeline")
    generate = next(n for n in starter["nodes"] if n["action"] == "Generate")
    expected = {"num_inference_steps": steps, "guidance_scale": guidance, "width": size, "height": size}
    for key, value in expected.items():
        assert generate["params"][key]["value"] == value, key
        assert generate["values"][key] == value, key
    assert loader["params"]["dtype"]["value"] == dtype
    assert loader["values"]["dtype"] == dtype
    assert MODULE_MAP["modules.DiffusersImage"] == original


def test_individually_resolved_lcm_node_uses_same_reviewed_defaults(contracts):
    result = resolve_operation(MODULE_MAP, contracts, {
        "pipelineClass": "LatentConsistencyModelPipeline", "task": "text_to_image",
        "operationId": "diffusion.generate_image",
    })
    assert result["params"]["guidance_scale"]["value"] == 8.5
    assert result["params"]["width"]["value"] == 512


def test_creating_another_model_does_not_rewrite_prior_draft(contracts):
    selected = {"pipelineClass": "FluxPipeline", "task": "text_to_image", "executionProfileId": "flux-schnell:direct"}
    draft = resolve_operation_starter(MODULE_MAP, contracts, selected)
    generate = next(n for n in draft["nodes"] if n["action"] == "Generate")
    generate["params"]["guidance_scale"]["value"] = 1.25
    snapshot = deepcopy(draft)
    resolve_operation_starter(MODULE_MAP, contracts, {**selected, "executionProfileId": "flux-krea:direct"})
    assert draft == snapshot


def test_all_public_image_starter_defaults_fit_their_declared_fields(contracts):
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES

    checked = 0
    for profile in DIFFUSERS_EXECUTION_PROFILES.values():
        if not profile.public or profile.loader_module != "modules.DiffusersImage":
            continue
        for task in profile.modes:
            starter = resolve_operation_starter(MODULE_MAP, contracts, {
                "pipelineClass": profile.pipeline_class, "task": task, "executionProfileId": profile.id,
            })
            for node in starter["nodes"]:
                for key in ("dtype", "width", "height", "num_inference_steps", "guidance_scale"):
                    field = node["params"].get(key)
                    if not field or field.get("hidden") or key not in node["values"]:
                        continue
                    value = node["values"][key]
                    identity = (profile.id, task, node["action"], key, value)
                    if isinstance(value, (int, float)):
                        if "min" in field:
                            assert value >= field["min"], identity
                        if "max" in field:
                            assert value <= field["max"], identity
                    if "options" in field:
                        assert value in field["options"], identity
                    checked += 1
    assert checked > 300
