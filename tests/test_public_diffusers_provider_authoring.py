"""Declared Expert authoring and mocked readiness; no model/native qualification."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock

import pytest

from modiff import ancillary_execution as ancillary
from modiff import diffusers_profiles as profiles
from modiff import optional_runtime_execution as execution
from modiff import optional_runtimes as runtimes
from modiff import runtime_overlays
from modiff import studio_execution_specs as specs


PROVIDER = runtimes.DIFFUSERS_FLUX3_RUNTIME_PROFILE_ID
PUBLIC = ("flux3-klein:official-provider:v1", "flux3-klein-base:official-provider:v1")
LEGACY = ("flux3-klein:modular", "flux3-klein-base:modular")


def forbidden(*_args, **_kwargs):
    pytest.fail("Authoring attempted an installed pipeline/model/runtime observation")


@pytest.mark.parametrize("identity", PUBLIC)
def test_normal_model_is_declared_before_optional_provider_and_has_complete_pup_offers(identity, monkeypatch):
    profile = profiles.DIFFUSERS_EXECUTION_PROFILES[identity]
    monkeypatch.setattr(runtime_overlays, "active_diffusers_provider", forbidden)
    monkeypatch.setattr(runtime_overlays, "base_diffusers_identity", forbidden)
    snapshots = []
    for state in ("base", "active", "repair_required", "restart_required"):
        monkeypatch.setenv("MODIFF_RUNTIME_OVERLAY_STATUS", state)
        declaration = profile.to_public_dict(platform_name="linux", machine="x86_64")
        assert declaration["public"] is True
        assert declaration["live_proof"] is False
        assert declaration["required_diffusers_provider"] == PROVIDER
        assert declaration["optionalRuntimeRequirement"]["requiredNow"] is True
        assert declaration["optionalRuntimeRequirement"]["state"] != "active"
        assert len(declaration["authoringRecipes"]) == 2
        for row in declaration["authoringRecipes"]:
            assert row["executionProfileId"] == identity
            assert row["task"] in {"text_to_image", "edit_image"}
            assert row["rawAuthoringRecipeId"] != row["authoringRecipeId"]
            assert [choice["label"] for choice in row["rawAuthoringRecipeChoices"]] == [
                "Raw wording: FP32", "Raw wording: BF16 + NF4"]
            assert row["rawAuthoringRecipeChoices"][0]["authoringRecipeId"] == row["rawAuthoringRecipeId"]
        capability = specs.studio_capability_definition(profile.model_type)
        assert capability["executionStatus"] == "expert_only"
        assert capability["qualificationStatus"] == "runtime-required"
        assert capability["autoEligible"] is True
        assert capability["qualifiedModes"] == []
        assert all(capability[key] is False for key in (
            "templateEligible", "galleryEligible", "liveProof"))
        snapshots.append((declaration, capability, [
            specs.studio_execution_spec_for_pair(profile.model_type, mode) for mode in profile.modes]))
    assert all(snapshot == snapshots[0] for snapshot in snapshots)


@pytest.mark.parametrize("identity", PUBLIC)
def test_declared_schema_and_normal_starter_do_not_need_optional_pipeline_class(identity, monkeypatch):
    import modules
    from modules.ModularDiffusers import modular_utils
    from modiff.operation_catalog import build_operation_catalog
    from modiff.operation_starters import resolve_operation_starter

    profile = profiles.DIFFUSERS_EXECUTION_PROFILES[identity]
    registry = modular_utils._get_registry_instance()
    installed = {owner: config for owner, config in registry.get_all().items()
                 if owner.__name__ != profile.pipeline_class}
    monkeypatch.setattr(registry, "get_all", lambda: installed)
    monkeypatch.setattr(modular_utils, "pipeline_class_from_model_type", forbidden)
    monkeypatch.setattr(runtime_overlays, "active_diffusers_provider", forbidden)
    metadata = modular_utils.get_model_type_metadata(profile.model_type)
    assert metadata["required_diffusers_provider"] == PROVIDER
    assert metadata["default_repo"] == profile.default_repo
    assert metadata["weight_formats"] == ["native", "fp8r", "nvfp4"]
    assert {"text_encoder", "vae_encoder", "denoise", "decoder"} <= set(metadata["node_params"])
    assert metadata["component_dtype_overrides"] == {"vae": "float32"}
    assert modular_utils.pipeline_authoring_identity(profile.model_type) == profile.model_type
    contracts, support = build_operation_catalog(modules.MODULE_MAP, [profile.to_public_dict()],
                                                  catalog_resolver=lambda: {})
    tasks = next(row["tasks"] for row in support if row["pipelineClass"] == profile.pipeline_class)
    assert {task["task"] for task in tasks if identity in task["executionProfileIds"]} == {
        "text_to_image", "edit_image"}
    for mode in profile.modes:
        selection = {"pipelineClass": profile.pipeline_class, "task": mode,
                     "executionProfileId": identity}
        starter = resolve_operation_starter(modules.MODULE_MAP, contracts, selection)
        assert starter["nodes"] and starter["edges"]
        loader = next(node for node in starter["nodes"] if node["action"] == "ModelsLoader")
        resolved, reason = profiles.resolve_execution_profiles_for_loader(
            loader['module'], loader['action'], loader['values'])
        assert reason is None and [selected.id for selected in resolved] == [identity]
        assert loader["values"]["repo_id"] == {"source": "hub", "value": profile.default_repo}
        if mode == "edit_image":
            assert any(node["action"] == "ImageEncode" for node in starter["nodes"])
    with pytest.raises(pytest.fail.Exception):
        modular_utils.require_modiff_node_contract(profile.model_type, "text_encoder")


def test_ordinary_loader_and_stage_refresh_use_declarations_without_optional_class(monkeypatch):
    from modules.ModularDiffusers import modular_utils
    from modules.ModularDiffusers.embeddings import EncodePrompt
    from modules.ModularDiffusers.loaders import ModelsLoader

    model = profiles.DIFFUSERS_EXECUTION_PROFILES[PUBLIC[0]].model_type
    monkeypatch.setattr(modular_utils, "pipeline_class_from_model_type", forbidden)
    loader = object.__new__(ModelsLoader)
    loader.node_id, loader.loader, loader._mm_models = 'public-authoring-loader-schema-unit', None, []
    loader._begin_pipeline_identity_refresh = Mock(return_value=1)
    loader._publish_pipeline_identity = Mock()
    loader.refresh_pipeline_identity({"model_type": model, "trust_remote_code": False}, None)
    assert loader._publish_pipeline_identity.call_args.kwargs["signal_value"] == model
    stage = object.__new__(EncodePrompt)
    stage._model_type, stage._pipeline_class = "", None
    stage.get_signal_value = lambda _name: model
    stage.send_node_definition = Mock()
    stage.update_node({}, None)
    assert stage._pipeline_class == model  # A string schema identity, never a fake class.
    assert "prompt" in stage.send_node_definition.call_args.args[0]


def test_explicit_private_spec_receipts_remain_distinct_and_unknown_collisions_refuse(monkeypatch):
    before = deepcopy(ancillary.PRIVATE_PARENTS)
    assert len(before) == 17 and len(ancillary.PUBLIC_PARENTS) == 20
    for old, new in zip(LEGACY, PUBLIC):
        public = profiles.DIFFUSERS_EXECUTION_PROFILES[new]
        for mode in public.modes:
            legacy = specs.studio_execution_spec_for_pair(public.model_type, mode, execution_profile_id=old)
            current = specs.studio_execution_spec_for_pair(public.model_type, mode)
            assert legacy["executionProfileId"] == old
            assert "requiredDiffusersProvider" not in legacy
            assert current["executionProfileId"] == new
            assert current["contentHash"] != legacy["contentHash"]
            assert specs.studio_execution_spec_for_pair(public.model_type, mode,
                content_hash=legacy["contentHash"]) == legacy
    assert ancillary.PRIVATE_PARENTS == before
    # An undeclared overlapping source cannot acquire default precedence.
    new = PUBLIC[0]
    definition = next(value for value in specs.STUDIO_EXECUTION_SPEC_DEFINITIONS.values()
                      if value["profile"]["id"] == new)
    with monkeypatch.context() as changed:
        for candidate in specs.STUDIO_EXECUTION_SPEC_DEFINITIONS.values():
            if candidate['profile']['id'] == new:
                changed.delitem(candidate['profile'], 'replaces_private_authoring_profile')
        assert specs.studio_execution_spec_for_pair(definition["modelType"], "text_to_image") is None


def test_source_replacement_retires_only_new_evaluation_authoring_and_requires_explicit_rebind(monkeypatch):
    from modiff import diffusers_evaluation_runtime as evaluation
    from modiff.operation_catalog import build_evaluation_authoring_catalog
    from modiff.operation_starters import _bind_execution_profile

    monkeypatch.setattr(evaluation, "selected_diffusers_evaluation_authoring_profile_ids", lambda: LEGACY)
    assert build_evaluation_authoring_catalog({}, catalog_resolver=forbidden) is None
    # This failure happens before legacy artifact observation or value mutation.
    loader = {"operation": {}, "params": {}, "values": {}}
    before = deepcopy(loader)
    with pytest.raises(ValueError, match="Rebind explicitly"):
        _bind_execution_profile(loader, "text_to_image", LEGACY[0])
    assert loader == before


def test_old_private_loader_retains_base_delivery_but_cannot_inherit_new_provider(monkeypatch):
    profile = profiles.DIFFUSERS_EXECUTION_PROFILES[LEGACY[0]]
    values = {"model_type": profile.pipeline_class, "execution_profile_id": profile.id,
              "repo_id": {"source": "hub", "value": profile.default_repo}}
    monkeypatch.setattr(runtime_overlays, "active_diffusers_provider", lambda: None)
    original = execution.loader_optional_runtime_requirement("modules.ModularDiffusers", "ModelsLoader", values,
                                                            catalog_resolver=forbidden)
    assert original["state"] == "base_satisfied"
    monkeypatch.setattr(runtime_overlays, "active_diffusers_provider", lambda: {"profileId": PROVIDER})
    current = execution.loader_optional_runtime_requirement("modules.ModularDiffusers", "ModelsLoader", values,
                                                           catalog_resolver=forbidden)
    assert current["reason"] == "execution_profile_rebind_required"
    assert execution.optional_runtime_requirement_blocks_execution(current)


def test_public_parent_choices_are_explicit_hash_covered_fixed_recipes():
    import modules
    for identity, declaration in ancillary.PUBLIC_PARENTS.items():
        parent = ancillary.resolve_parent(modules.MODULE_MAP, identity)
        assert parent["id"] == identity
        child = parent["ancillaryOperations"][0]
        literals = {(role, field): value for role, field, value in child["initialValues"]}
        assert literals[("provider", "device")] == "cuda:0"
        assert literals[("proposal", "max_input_tokens")] == 4096
        assert literals[("provider", "dtype")] == (
            "bfloat16" if declaration["providerRecipe"] == "bf16-nf4" else "float32")
        assert literals[("provider", "quantization_mode")] == (
            "bnb_4bit" if declaration["providerRecipe"] == "bf16-nf4" else "none")
        assert child.get("layoutPolicy") == declaration.get("layoutPolicy")
        assert (child["bindings"]["owner"]["referenceSource"] is not None) is (
            declaration["task"] == "edit_image")
        assert child["contentHash"] == "studio-ancillary-v1-" + specs._hash_string(specs._stable_json(
            {key: value for key, value in child.items() if key != "contentHash"}))


def test_no_unqualified_provider_action_or_model_qualification_is_promoted():
    provider = runtimes.OPTIONAL_RUNTIME_PROFILES[PROVIDER]
    assert provider.provides_diffusers is True
    for target in provider.target_contracts:
        # These are package/activation qualifications, never live model proof.
        qualified = (target.platform, target.machine) in {('linux', 'x86_64'), ('windows', 'x86_64')}
        assert target.cutover_ready is qualified
        assert target.install_action_available is qualified
        assert target.activation_available is qualified
    for identity in PUBLIC:
        profile = profiles.DIFFUSERS_EXECUTION_PROFILES[identity]
        assert profile.component_lora_declaration.profile_id == identity
        assert profile.adapter_authoring is not None and profile.live_proof is False
        assert profiles.execution_profiles_for_execution(profile.model_type) == (profile,)
        with pytest.raises(ValueError, match="exact declared"):
            replace(profile, required_diffusers_provider="undeclared")


@pytest.mark.parametrize("identity", PUBLIC)
def test_mandatory_provider_cannot_fall_back_to_base_on_any_declared_target(identity):
    profile = profiles.DIFFUSERS_EXECUTION_PROFILES[identity]
    for platform_name, machine, _delivery in profiles.OPTIONAL_RUNTIME_PLATFORM_DELIVERIES:
        requirement = profiles.optional_runtime_requirement_for_profiles(
            (profile,), platform_name=platform_name, machine=machine)
        assert requirement["delivery"] == "optional_overlay"
        assert requirement["requiredNow"] is True
        assert requirement["profileIds"] == [PROVIDER]
        assert requirement["state"] == "unavailable"
        assert requirement["reason"] == "optional_runtime_status_required"
    with pytest.raises(ValueError, match="exact declared overlay"):
        replace(profile, optional_runtime_platform_deliveries=(("macos", "arm64", "base"),))
