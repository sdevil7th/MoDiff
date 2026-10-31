"""No-model regression tests: metadata and mocked process identity only."""
from copy import deepcopy
from dataclasses import replace

import pytest

from modiff import diffusers_profiles as profiles, optional_runtime_execution as execution
from modiff import optional_runtimes as runtimes, runtime_overlays, studio_execution_specs as specs

PID = runtimes.DIFFUSERS_FLUX3_RUNTIME_PROFILE_ID
PUBLIC_ID = "source-provider-test:public"


def public_profile():
    return replace(profiles.DIFFUSERS_EXECUTION_PROFILES["flux3-klein:modular"],
        id=PUBLIC_ID, public=True, component_lora_declaration=None, adapter_authoring=None,
        required_diffusers_provider=PID, optional_runtime_profiles=(PID,),
        optional_runtime_delivery=profiles.OPTIONAL_RUNTIME_DELIVERY_OVERLAY,
        optional_runtime_platform_deliveries=())


def qualify_source_in_memory(monkeypatch, qualified):
    original = runtimes.OPTIONAL_RUNTIME_PROFILES[PID]
    profile = replace(original, target_contracts=tuple(replace(target,
        contract_state="qualified" if qualified else "candidate_unqualified", cutover_ready=qualified,
        install_action_available=qualified, activation_available=qualified)
        for target in original.target_contracts))
    monkeypatch.setattr(runtimes, "OPTIONAL_RUNTIME_PROFILES", {**runtimes.OPTIONAL_RUNTIME_PROFILES, PID: profile})
    monkeypatch.setattr(execution, "OPTIONAL_RUNTIME_PROFILES", runtimes.OPTIONAL_RUNTIME_PROFILES)
    return profile


def active_catalog(profile):
    return {"schemaVersion": 1, "overlay": {"processLoadStatus": "active"}, "profiles": [{
        **profile.to_spec_dict(), "specDigest": profile.spec_digest,
        "status": "present_unqualified", "overlayStatus": "active",
        "contractState": "qualified", "cutoverReady": True, "baseIncluded": False}]}


@pytest.mark.parametrize("state,expected", [
    ("none", False), ("E_only", False), ("provider", True),
    ("unqualified_target", False), ("wrong_provider", False),
    ("provider_drift", False), ("base_drift", False),
])
def test_observed_execution_requires_exact_sealed_identity_not_env_or_catalog(monkeypatch, state, expected):
    contract = qualify_source_in_memory(monkeypatch, state != "unqualified_target")
    profile = public_profile()
    selected = {"profileId": PID, "specDigest": contract.spec_digest,
                "distribution": "diffusers", "version": contract.packages[0].version}
    if state in {"none", "E_only"}:
        selected = None
    elif state == "wrong_provider":
        selected["profileId"] = "different-provider"
    elif state == "provider_drift":
        selected["specDigest"] = "sha256:" + "0" * 64
    monkeypatch.setattr(runtime_overlays, "active_diffusers_provider", lambda: selected)
    def base_identity():
        if state == "base_drift":
            raise RuntimeError("actual base identity drift")
        return {"originalBase": "still exact"}
    monkeypatch.setattr(runtime_overlays, "base_diffusers_identity", base_identity)
    monkeypatch.setenv("MODIFF_RUNTIME_OVERLAY_STATUS", "active")
    if state == "E_only":
        monkeypatch.setenv("MODIFF_DIFFUSERS_EVALUATION_PROFILE", "old-private-selector")
    result = execution.optional_runtime_requirement_for_profiles((profile,),
        catalog_resolver=lambda: active_catalog(contract))
    assert (result["state"] == "active") is expected
    assert result["requiredNow"] is True
    assert execution.optional_runtime_requirement_blocks_execution(result) is not expected


def test_declarative_public_inventory_and_spec_hash_do_not_observe_process(monkeypatch):
    profile = public_profile()
    monkeypatch.setitem(profiles.DIFFUSERS_EXECUTION_PROFILES, PUBLIC_ID, profile)
    definition = deepcopy(specs.STUDIO_EXECUTION_SPEC_DEFINITIONS[
        "flux3-klein:modular:text-to-image:v1"])
    definition["profile"].update(id=PUBLIC_ID, public=True, required_diffusers_provider=PID)
    def forbidden(*_args):
        pytest.fail("declarative inventory consulted process runtime authority")
    monkeypatch.setattr(runtime_overlays, "active_diffusers_provider", forbidden)
    monkeypatch.setattr(runtime_overlays, "base_diffusers_identity", forbidden)
    snapshots = []
    for env in ("base", "active", "repair_required", "restart_required"):
        monkeypatch.setenv("MODIFF_RUNTIME_OVERLAY_STATUS", env)
        declarations = profiles.public_execution_profiles(observe_optional_runtime=False,
            platform_name="linux", machine="x86_64")
        item = next(row for row in declarations if row["id"] == PUBLIC_ID)
        assert item["optionalRuntimeRequirement"]["requiredNow"] is True
        assert item["optionalRuntimeRequirement"]["state"] != "active"
        specification = specs._public_spec("source-provider-test:text-to-image:v1", definition)
        assert specification["requiredDiffusersProvider"] == PID
        snapshots.append((declarations, specification))
    assert all(value == snapshots[0] for value in snapshots)


def test_old_declarations_omit_new_field_and_remain_source_only(monkeypatch):
    for profile in profiles.DIFFUSERS_EXECUTION_PROFILES.values():
        if profile.required_diffusers_provider is None:
            assert "required_diffusers_provider" not in profile.to_public_dict()
    with pytest.raises(ValueError, match="exact declared"):
        replace(public_profile(), required_diffusers_provider="undeclared")
    with pytest.raises(ValueError, match="exact declared"):
        replace(public_profile(), optional_runtime_delivery=profiles.OPTIONAL_RUNTIME_DELIVERY_BASE)
