"""Static selected stage claims do not promote runtime or model qualification."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import modiff
import pytest

from modiff import image_prototyping_readiness as readiness
from modiff import upstream_coverage as coverage
from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.studio_execution_specs import studio_execution_spec_for_pair


ROOT = Path(modiff.__file__).resolve().parents[1]
TEMPLATE_IDS = (
    "flux_fill_inpaint", "flux_fill_outpaint", "flux_control_canny",
    "flux_depth_control", "flux_redux_edit", "flux_redux_multi_reference",
)


@pytest.fixture(scope="module")
def templates():
    rows = coverage._load_public_templates(ROOT / "web/assets/studio-templates.js")
    return {row["id"]: row for row in rows}


def check(selection, **kwargs):
    import builtins

    original = builtins.__import__

    def no_model_or_registry(name, *args, **kw):
        if name.split(".", 1)[0] in {"modules", "torch", "diffusers", "transformers", "peft"}:
            raise AssertionError("Static coverage imported a registry or model library")
        return original(name, *args, **kw)

    with patch("builtins.__import__", no_model_or_registry), patch(
        "socket.socket.connect", side_effect=AssertionError("network")
    ):
        return coverage._template_execution_coverage(selection, **kwargs)


@pytest.mark.parametrize("identity", TEMPLATE_IDS)
def test_actual_six_truthful_stage_claims_keep_historical_evidence_separate(templates, identity):
    selection = deepcopy(templates[identity]["executionSelection"])
    assert selection["implementation"] == "pipeline_stages"
    result = check(selection)
    assert result["executionSelection"] == selection
    assert result["executionRecipeSource"] == "backend_operation_starter"
    assert result["canonicalWorkflowRole"] == "historical_creator_recipe_reference"
    assert result["galleryExecutionCompatibility"] == "historical_reference_requires_new_execution_evidence"
    assert set(result) == {
        "executionSelection", "executionRecipeSource", "canonicalWorkflowRole", "galleryExecutionCompatibility"
    }
    profile = DIFFUSERS_EXECUTION_PROFILES[selection["executionProfileId"]]
    before = profile.to_public_dict()
    check(selection)
    assert profile.to_public_dict() == before


@pytest.mark.parametrize("identity", TEMPLATE_IDS)
def test_existing_whole_pipeline_compatibility_claim_stays_accepted(templates, identity):
    selection = {**templates[identity]["executionSelection"], "implementation": "whole_pipeline"}
    assert check(selection)["executionSelection"] == selection


def test_native_claim_and_absent_selection_keep_existing_behavior(templates):
    selection = templates["flux_schnell_text_to_image"]["executionSelection"]
    assert selection["implementation"] == "native_stages"
    assert check(selection)["executionSelection"] == selection
    assert check(None) == {}
    with pytest.raises(coverage.UpstreamCoverageError):
        check({**selection, "implementation": "pipeline_stages"})


@pytest.mark.parametrize("corruption", [
    "foreign_profile", "canny_depth_binding", "foreign_mode", "task_binding_disagreement",
    "whole_sibling", "missing_binding", "malformed_binding", "native_label", "unknown_label",
])
def test_stage_claim_requires_exact_selected_profile_task_and_binding(templates, corruption):
    selection = deepcopy(templates["flux_control_canny"]["executionSelection"])
    if corruption == "foreign_profile":
        selection["executionProfileId"] = "flux-fill:direct"
    elif corruption == "canny_depth_binding":
        selection["bindingSpec"]["modelType"] = "FluxDepthPipeline"
    elif corruption == "foreign_mode":
        selection["task"] = "text_to_image"
    elif corruption == "task_binding_disagreement":
        selection = deepcopy(templates["flux_fill_inpaint"]["executionSelection"])
        selection["bindingSpec"]["mode"] = "outpaint"
    elif corruption == "whole_sibling":
        selection["task"] = selection["bindingSpec"]["mode"] = "image_to_image"
    elif corruption == "missing_binding":
        selection.pop("bindingSpec")
    elif corruption == "malformed_binding":
        selection["bindingSpec"] = []
    else:
        selection["implementation"] = "native_stages" if corruption == "native_label" else "not_reviewed"
    with pytest.raises(coverage.UpstreamCoverageError):
        check(selection)


@pytest.mark.parametrize("corruption", [
    "missing_stage", "missing_geometry", "foreign_profile", "foreign_primary", "mutable_pin",
    "wrong_pin", "duplicate_supplier", "missing_preview", "missing_spec",
])
def test_stage_claim_reuses_strict_static_declaration_controls(templates, monkeypatch, corruption):
    selection = deepcopy(templates["flux_fill_inpaint"]["executionSelection"])
    spec = studio_execution_spec_for_pair("FluxFillPipeline", "inpaint")
    if corruption == "missing_stage":
        spec["roles"] = [row for row in spec["roles"] if row[1] != "modules.DiffusersImage.Denoise"]
    elif corruption == "missing_geometry":
        spec["edges"] = [row for row in spec["edges"] if row[1] != "width_out"]
    elif corruption == "foreign_profile":
        spec["executionProfileId"] = "flux-redux:direct"
    elif corruption == "foreign_primary":
        spec["defaultRepo"] = "foreign/model"
    elif corruption in {"mutable_pin", "wrong_pin"}:
        from modiff import image_stage_contract
        pin = catalog_repository_pin(DIFFUSERS_EXECUTION_PROFILES["flux-fill:direct"].default_repo)
        monkeypatch.setattr(image_stage_contract, "catalog_repository_pin", lambda *_args, **_kwargs: {
            **pin, "revision": "main" if corruption == "mutable_pin" else "a" * 40,
        })
    elif corruption == "duplicate_supplier":
        spec["edges"] = [*spec["edges"], ("imageEncode", "encoded_inputs", "decode", "width")]
    elif corruption == "missing_preview":
        spec["edges"] = [row for row in spec["edges"] if not (row[0] == "decode" and row[1] == "images")]
    else:
        spec = None
    monkeypatch.setattr(coverage, "studio_execution_spec_for_pair", lambda *_args, **_kwargs: spec)
    with pytest.raises(coverage.UpstreamCoverageError):
        check(selection)


@pytest.mark.parametrize("corruption", ["foreign_repo", "wrong_revision", "mutable_revision", "local", "malformed"])
def test_explicit_stage_artifact_must_be_the_immutable_primary(templates, corruption):
    selection = templates["flux_fill_inpaint"]["executionSelection"]
    profile = DIFFUSERS_EXECUTION_PROFILES["flux-fill:direct"]
    pin = catalog_repository_pin(profile.default_repo, model_type=profile.model_type)
    artifact = {"source": "hub", "value": profile.default_repo, "revision": pin["revision"]}
    if corruption == "foreign_repo":
        artifact["value"] = "foreign/model"
    elif corruption in {"wrong_revision", "mutable_revision"}:
        artifact["revision"] = "a" * 40 if corruption == "wrong_revision" else "main"
    elif corruption == "local":
        artifact["source"] = "local"
    else:
        artifact = []
    with pytest.raises(coverage.UpstreamCoverageError):
        check(selection, artifact=artifact)


def test_exact_primary_artifact_is_accepted_and_nonstaged_claims_remain_unchanged(templates):
    selection = templates["flux_fill_inpaint"]["executionSelection"]
    profile = DIFFUSERS_EXECUTION_PROFILES["flux-fill:direct"]
    pin = catalog_repository_pin(profile.default_repo, model_type=profile.model_type)
    artifact = {"source": "hub", "value": profile.default_repo, "revision": pin["revision"]}
    assert check(selection, artifact=artifact)["executionSelection"] == selection
    assert check({**selection, "implementation": "whole_pipeline"}, artifact=[])["executionSelection"][
        "implementation"] == "whole_pipeline"


def test_readiness_alias_keeps_the_same_shared_predicate():
    from modiff.image_stage_contract import declared_backend_image_stages
    assert readiness._declared_backend_image_stages is declared_backend_image_stages
