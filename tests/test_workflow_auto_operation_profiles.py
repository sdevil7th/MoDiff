"""Native recipe selection must survive the form-to-graph resource boundary."""

from copy import deepcopy
from dataclasses import replace

import pytest

from modiff import auto_resource, workflow_auto_resource
from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
from modiff.model_artifact_catalog import require_catalog_revision
from modiff.optional_runtime_execution import assert_optional_runtime_ready, OptionalRuntimeExecutionBlocked
from modiff.optional_runtimes import TRANSFORMERS_MAIN_PEFT_QUANTO_RUNTIME_PROFILE_ID


GIB = 1024 ** 3


def native_flux_graph(repository, *, execution_profile_id=None, sequence_length=512):
    def node(action, **values):
        return {"module": "modules.ModularDiffusers", "action": action,
                "params": {key: value if isinstance(value, dict) and "sourceId" in value else {"value": value}
                           for key, value in values.items()}}

    nodes = {
        "load": node("ModelsLoader", model_type="FluxModularPipeline",
                     repo_id={"source": "hub", "value": repository}, revision=require_catalog_revision(repository),
                     dtype="bfloat16", device="cuda:0", offload_mode="model_cpu", auto_offload=True),
        "encode": node("EncodePrompt", text_encoders={"sourceId": "load", "sourceKey": "text_encoders"},
                       prompt="A coastal landscape", max_sequence_length=sequence_length),
        "denoise": node("Denoise", unet={"sourceId": "load", "sourceKey": "unet_out"},
                        embeddings={"sourceId": "encode", "sourceKey": "embeddings"},
                        scheduler={"sourceId": "load", "sourceKey": "scheduler"},
                        width=1024, height=1024, num_inference_steps=4, guidance_scale=0),
        "decode": node("DecodeLatents", vae={"sourceId": "load", "sourceKey": "vae_out"},
                       latents={"sourceId": "denoise", "sourceKey": "latents"}, width=1024, height=1024),
    }
    if execution_profile_id is not None:
        nodes["load"]["params"]["execution_profile_id"] = {"value": execution_profile_id}
    return {"nodes": nodes, "paths": [list(nodes)]}


@pytest.fixture
def actual_planners(monkeypatch, tmp_path):
    # Exercise the real profile, artifact-revision and candidate matching paths;
    # isolate only the installed-cache observation, without loading any models.
    monkeypatch.setattr(auto_resource, "_artifact_cache_status", lambda repo, models: {
        "installed": True, "complete": True, "repo": repo})
    hardware = {
        "platform": "linux", "architecture": "x86_64",
        "accelerator": {"kind": "cuda", "totalBytes": 32 * GIB, "freeBytes": 28 * GIB},
        "systemMemory": {"totalBytes": 64 * GIB, "availableBytes": 60 * GIB},
        "offloadDisk": {"freeBytes": 200 * GIB},
    }
    requests = []

    def recipe(payload, **kwargs):
        requests.append(deepcopy(payload))
        return auto_resource.build_auto_resource_plan({**payload, "hardwareOverride": hardware}, **kwargs)

    def workflow(graph):
        return workflow_auto_resource.build_workflow_auto_plan(
            graph, runtime_fingerprint={}, local_models=[], data_dir=str(tmp_path),
            hardware=hardware, plan_recipe=recipe)

    def form(values):
        return auto_resource.build_auto_resource_plan(
            {"form": values, "hardwareOverride": hardware}, runtime_fingerprint={}, local_models=[],
            data_dir=str(tmp_path))

    return workflow, form, requests


def test_real_native_schnell_auto_preserves_selected_profile_and_512_tokens(actual_planners):
    workflow, _, requests = actual_planners
    profile = DIFFUSERS_EXECUTION_PROFILES["flux-schnell:modular"]
    graph = native_flux_graph(profile.default_repo)
    before = deepcopy(graph)
    result = workflow(graph)
    assert result["canAutoRun"], result["issues"]
    assert requests[-1]["form"]["executionProfileId"] == profile.id
    assert result["loaders"][0]["settings"]["executionProfileId"] == profile.id
    assert result["loaders"][0]["settings"]["maxSequenceLength"] == 512
    assert result["resolvedFields"]["encode"]["max_sequence_length"] == 512
    assert result["loaders"][0]["proofStatus"] == "declared_safe"
    assert not any(patch["field"] == "max_sequence_length" for patch in result["patches"])
    assert graph == before


def native_schnell_form(**overrides):
    profile = DIFFUSERS_EXECUTION_PROFILES["flux-schnell:modular"]
    return {"modelType": profile.model_type, "modelRepo": profile.default_repo, "mode": "text_to_image",
            "executionProfileId": profile.id, "dtype": "bfloat16", "width": 1024, "height": 1024,
            "steps": 4, "guidanceScale": 0, "maxSequenceLength": 512, **overrides}


def test_native_recipe_candidate_has_actual_target_and_no_standard_graph_contract(actual_planners):
    _, form, _ = actual_planners
    result = form(native_schnell_form())
    candidate = result["selectedCandidate"]
    assert candidate["executionProfileId"] == "flux-schnell:modular"
    assert (candidate["loaderModule"], candidate["loaderAction"], candidate["executionPath"],
            candidate["pipelineClass"]) == (
        "modules.ModularDiffusers", "ModelsLoader", "modular-diffusers", "FluxModularPipeline")
    assert candidate["generation"]["maxSequenceLength"] == 512
    assert candidate["requirements"]["memorySemantics"] == "machine_capacity"
    assert candidate["proof"]["status"] == "declared_safe"
    assert "studioExecutionSpecContract" not in candidate
    assert result["optionalRuntimeRequirement"]["executionProfileIds"] == ["flux-schnell:modular"]
    assert candidate["requirements"]["supportedOffloadModes"] == list(
        DIFFUSERS_EXECUTION_PROFILES["flux-schnell:modular"].supported_offload_modes)


def test_legacy_form_without_recipe_selection_keeps_its_original_loader(actual_planners):
    _, form, _ = actual_planners
    values = native_schnell_form()
    values.pop("executionProfileId")
    result = form(values)
    candidate = result["selectedCandidate"]
    assert candidate["executionProfileId"] == "flux-schnell:direct"
    assert candidate["loaderAction"] == "LoadPipeline"
    assert candidate["pipelineClass"] == "FluxPipeline"
    assert candidate["studioExecutionSpecContract"]["executionProfileId"] == "flux-schnell:direct"


@pytest.mark.parametrize("selection", ["unknown:modular", "flux-dev:modular", " flux-schnell:modular", "", False, {}])
def test_unknown_invalid_or_wrong_model_recipe_selection_is_not_replaced_by_legacy_auto(actual_planners, selection):
    _, form, _ = actual_planners
    result = form(native_schnell_form(executionProfileId=selection))
    assert not result["canAutoRun"]
    assert result["selectedCandidate"] is None
    assert all(candidate["exactPairDeclared"] is False for candidate in result["candidates"])


def test_recipe_selection_does_not_grant_a_new_task(actual_planners):
    _, form, _ = actual_planners
    result = form(native_schnell_form(mode="image_to_image"))
    assert not result["canAutoRun"]


@pytest.mark.parametrize("override", [
    {"execution_profile_id": "flux-krea:modular"},
    {"execution_profile_id": "flux-schnell:direct"},
    {"revision": "0" * 40},
    {"repo_id": {"source": "hub", "value": "unreviewed/model"}},
])
def test_native_graph_wrong_profile_repository_or_revision_remains_blocked(actual_planners, override):
    workflow, _, _ = actual_planners
    graph = native_flux_graph(DIFFUSERS_EXECUTION_PROFILES["flux-schnell:modular"].default_repo)
    for field, value in override.items():
        graph["nodes"]["load"]["params"][field] = {"value": value}
    assert not workflow(graph)["canAutoRun"]


def test_other_reviewed_flux_variant_uses_its_own_native_recipe(actual_planners):
    workflow, _, requests = actual_planners
    profile = DIFFUSERS_EXECUTION_PROFILES["flux-krea:modular"]
    graph = native_flux_graph(profile.default_repo)
    graph["nodes"]["denoise"]["params"]["num_inference_steps"]["value"] = 30
    graph["nodes"]["denoise"]["params"]["guidance_scale"]["value"] = 3.5
    result = workflow(graph)
    assert result["canAutoRun"], result["issues"]
    assert requests[-1]["form"]["executionProfileId"] == profile.id
    assert result["loaders"][0]["settings"]["steps"] == 30
    assert result["loaders"][0]["repository"] == profile.default_repo


def test_legacy_success_evidence_cannot_be_transferred_to_native_recipe(actual_planners):
    _, form, _ = actual_planners
    values = native_schnell_form()
    native_plan = form(values)
    values.pop("executionProfileId")
    direct = form(values)["selectedCandidate"]
    native = native_plan["selectedCandidate"]
    signature = auto_resource._candidate_history_signature(direct, hardware=native_plan["hardware"])
    result = auto_resource._apply_history_to_candidates([native], hardware=native_plan["hardware"], history={
        "version": 2, "entries": {"old-direct": {
            "signature": signature, "candidate": direct, "successCount": 1, "lastSuccessAt": 1}}})[0]
    assert result["proof"]["status"] == "declared_safe"
    assert result["successHistory"] is None


def test_exact_recipe_optional_runtime_gate_is_preserved(actual_planners, monkeypatch):
    _, form, _ = actual_planners
    profile = DIFFUSERS_EXECUTION_PROFILES["flux-schnell:modular"]
    monkeypatch.setitem(DIFFUSERS_EXECUTION_PROFILES, profile.id, replace(
        profile, optional_runtime_profiles=(TRANSFORMERS_MAIN_PEFT_QUANTO_RUNTIME_PROFILE_ID,),
        optional_runtime_delivery="optional_overlay", optional_runtime_platform_deliveries=()))
    monkeypatch.setenv("MODIFF_RUNTIME_OVERLAY_STATUS", "base")
    monkeypatch.setattr("modiff.optional_runtime_execution.public_optional_runtime_catalog", lambda: {})
    result = form(native_schnell_form())
    assert result["optionalRuntimeRequirement"]["executionProfileIds"] == [profile.id]
    assert result["optionalRuntimeRequirement"]["requiredNow"]
    assert result["optionalRuntimeRequirement"]["state"] == "unavailable"
    # Resource candidate availability and actual package execution readiness
    # remain separate contracts; exact-recipe selection must retain the gate.
    with pytest.raises(OptionalRuntimeExecutionBlocked):
        assert_optional_runtime_ready(result["optionalRuntimeRequirement"])


@pytest.mark.parametrize("process_state", ["restart_required", "repair_required", "busy_recovery_only"])
def test_exact_recipe_optional_runtime_keeps_process_recovery_gate(actual_planners, monkeypatch, process_state):
    _, form, _ = actual_planners
    profile = DIFFUSERS_EXECUTION_PROFILES["flux-schnell:modular"]
    monkeypatch.setitem(DIFFUSERS_EXECUTION_PROFILES, profile.id, replace(
        profile, optional_runtime_profiles=(TRANSFORMERS_MAIN_PEFT_QUANTO_RUNTIME_PROFILE_ID,),
        optional_runtime_delivery="optional_overlay", optional_runtime_platform_deliveries=()))
    monkeypatch.setenv("MODIFF_RUNTIME_OVERLAY_STATUS", process_state)

    def no_catalog_probe():
        raise AssertionError("Process recovery must block before installed-catalog inspection")

    monkeypatch.setattr("modiff.optional_runtime_execution.public_optional_runtime_catalog", no_catalog_probe)
    result = form(native_schnell_form())
    requirement = result["optionalRuntimeRequirement"]
    assert requirement["executionProfileIds"] == [profile.id]
    assert requirement["requiredNow"]
    assert requirement["state"] == process_state
    with pytest.raises(OptionalRuntimeExecutionBlocked):
        assert_optional_runtime_ready(requirement)
