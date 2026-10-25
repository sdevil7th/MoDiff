"""No-weight contracts for one explicitly selected provider/NF4 closure."""

from copy import deepcopy
from dataclasses import replace

import pytest

from modiff import optional_runtimes as runtimes, runtime_overlays


COMPOSITE_ID = "huggingface-diffusers-flux3-7ecb1e1-bitsandbytes-0.50.0"
PROVIDER_ID = runtimes.DIFFUSERS_FLUX3_RUNTIME_PROFILE_ID
BNB_ID = runtimes.TRANSFORMERS_MAIN_PEFT_BITSANDBYTES_RUNTIME_PROFILE_ID


def _profile(profile_id):
    profile = runtimes.OPTIONAL_RUNTIME_PROFILES.get(profile_id)
    assert profile is not None, "The explicit Diffusers/bitsandbytes closure is not declared."
    return profile


def _qualified(profile):
    # Synthetic target authority exercises the predicate, not source qualification.
    return replace(profile, target_contracts=tuple(
        replace(target, contract_state="qualified", cutover_ready=True,
                install_action_available=True, activation_available=True)
        for target in profile.target_contracts
    ))


def _qualified_context(monkeypatch):
    bare = _qualified(_profile(PROVIDER_ID))
    bnb = _profile(BNB_ID)
    # Build an in-memory source declaration so the predicate's causal RED is
    # its old identity-only comparison, independently of the new inventory row.
    composite = replace(
        bare, id=COMPOSITE_ID, label="Synthetic qualified composite",
        packages=(*bare.packages, *bnb.packages),
        artifact_locks=(*bare.artifact_locks, *bnb.artifact_locks),
        satisfies_profiles=((bare.id, bare.spec_digest), (bnb.id, bnb.spec_digest)),
    )
    monkeypatch.setattr(runtimes, "OPTIONAL_RUNTIME_PROFILES", {
        **runtimes.OPTIONAL_RUNTIME_PROFILES, bare.id: bare, composite.id: composite,
    })
    selected = {
        "profileId": composite.id, "specDigest": composite.spec_digest,
        "distribution": "diffusers", "version": bare.packages[0].version,
    }
    monkeypatch.setattr(runtime_overlays, "active_diffusers_provider", lambda: selected)
    monkeypatch.setattr(runtime_overlays, "base_diffusers_identity", lambda: {"originalBase": "exact"})
    return bare, composite, selected


def test_explicit_composite_declares_only_exact_provider_and_auxiliary_packages():
    composite = _profile(COMPOSITE_ID)
    bare, bnb = _profile(PROVIDER_ID), _profile(BNB_ID)
    assert runtimes.DIFFUSERS_FLUX3_BITSANDBYTES_RUNTIME_PROFILE_ID == COMPOSITE_ID
    assert composite.provides_diffusers is True
    assert composite.packages == (*bare.packages, *bnb.packages)
    assert {item.distribution for item in composite.packages} == {"diffusers", "bitsandbytes"}
    assert composite.base_packages == bare.base_packages
    assert {item.distribution for item in bnb.base_packages}.issubset(
        {item.distribution for item in composite.base_packages}
    )
    assert composite.source_builds == bare.source_builds
    assert composite.artifact_locks == (*bare.artifact_locks, *bnb.artifact_locks)
    assert composite.satisfies_profiles == ((bare.id, bare.spec_digest), (bnb.id, bnb.spec_digest))
    assert composite.source_builds[0]["outputWheel"]["sha256"] == (
        "59c8eb8aaaa25fe5f912e09c48bf66f7e41655c468a7ec47f56da93d8ca40dbd"
    )
    windows_bnb = [row for row in composite.artifact_locks
                   if row["distribution"] == "bitsandbytes" and row["platform"] == "windows"]
    assert len(windows_bnb) == 1
    assert windows_bnb[0]["sha256"] == "7f9ebf51d76a00341711cdd557af97555274b5aad3444975ff9927866a18bc11"
    assert len(composite.target_contracts) == 6
    assert composite.contract_state == "candidate_unqualified"
    assert (composite.cutover_ready, composite.install_action_available, composite.activation_available) == (
        False, False, False,
    )
    for target in composite.target_contracts:
        assert target.contract_state == "candidate_unqualified"
        assert (target.cutover_ready, target.install_action_available, target.activation_available) == (
            False, False, False,
        )


def test_qualification_workload_is_closed_and_omitted_from_existing_profiles():
    composite = _profile(COMPOSITE_ID)
    assert composite.qualification_workload == runtimes.QUALIFICATION_WORKLOAD_CUDA_NF4
    assert composite.to_spec_dict()["qualificationWorkload"] == "cuda-nf4-v1"
    declared = []
    for profile in runtimes.OPTIONAL_RUNTIME_PROFILES.values():
        if profile.qualification_workload is not None:
            declared.append(profile.id)
        else:
            assert "qualificationWorkload" not in profile.to_spec_dict()
    assert declared == [composite.id]
    for invalid in ("unreviewed-workload", "", True, 1):
        with pytest.raises(ValueError, match="supported declared workload"):
            replace(composite, qualification_workload=invalid)
    for profile_id in (PROVIDER_ID, BNB_ID):
        with pytest.raises(ValueError, match="explicit Diffusers/bitsandbytes"):
            replace(_profile(profile_id), qualification_workload="cuda-nf4-v1")


def test_current_qualified_composite_satisfies_exact_bare_provider(monkeypatch):
    bare, _composite, _selected = _qualified_context(monkeypatch)
    # Causal RED: the former identity-only predicate rejects the distinct closure.
    assert runtime_overlays.required_diffusers_provider_available(bare.id) is True


@pytest.mark.parametrize("change", [
    "missing_satisfaction", "stale_satisfaction", "other_satisfaction", "selected_spec",
    "selected_profile", "selected_version", "selected_distribution", "bare_target",
    "composite_target", "source_lock", "package_contract", "base_drift", "base_absent",
])
def test_composite_provider_refuses_unbound_or_unqualified_authority(monkeypatch, change):
    bare, composite, selected = _qualified_context(monkeypatch)
    if change == "missing_satisfaction":
        composite = replace(composite, satisfies_profiles=())
    elif change == "stale_satisfaction":
        composite = replace(composite, satisfies_profiles=((bare.id, "sha256:" + "0" * 64),))
    elif change == "other_satisfaction":
        composite = replace(composite, satisfies_profiles=((BNB_ID, _profile(BNB_ID).spec_digest),))
    elif change == "selected_spec":
        selected["specDigest"] = "sha256:" + "0" * 64
    elif change == "selected_profile":
        selected["profileId"] = "undeclared-provider"
    elif change == "selected_version":
        selected["version"] = "0.41.0"
    elif change == "selected_distribution":
        selected["distribution"] = "different-root"
    elif change in {"bare_target", "composite_target"}:
        target_profile = bare if change == "bare_target" else composite
        candidate = replace(target_profile, target_contracts=tuple(
            replace(target, contract_state="candidate_unqualified", cutover_ready=False,
                    install_action_available=False, activation_available=False)
            for target in target_profile.target_contracts
        ))
        if change == "bare_target":
            bare = candidate
            composite = replace(composite, satisfies_profiles=((bare.id, bare.spec_digest),))
        else:
            composite = candidate
    elif change == "source_lock":
        source = deepcopy(composite.source_builds[0])
        source["outputWheel"]["sha256"] = "0" * 64
        composite = replace(composite, source_builds=(source,))
    elif change == "package_contract":
        root = replace(composite.packages[0], required_symbols=("DifferentAPI",))
        composite = replace(composite, packages=(root, *composite.packages[1:]))
    elif change == "base_drift":
        def drift():
            raise RuntimeError("original base installation drifted")
        monkeypatch.setattr(runtime_overlays, "base_diffusers_identity", drift)
    elif change == "base_absent":
        monkeypatch.setattr(runtime_overlays, "base_diffusers_identity", lambda: None)
    if change not in {"selected_spec", "selected_profile"}:
        selected["specDigest"] = composite.spec_digest
    monkeypatch.setattr(runtimes, "OPTIONAL_RUNTIME_PROFILES", {
        **runtimes.OPTIONAL_RUNTIME_PROFILES, bare.id: bare, composite.id: composite,
    })
    assert runtime_overlays.required_diffusers_provider_available(bare.id) is False


def test_direct_bare_provider_and_absent_requirement_are_unchanged(monkeypatch):
    bare, _composite, selected = _qualified_context(monkeypatch)
    selected.update(profileId=bare.id, specDigest=bare.spec_digest)
    assert runtime_overlays.required_diffusers_provider_available(bare.id) is True
    assert runtime_overlays.required_diffusers_provider_available(None) is True
    assert runtime_overlays.required_diffusers_provider_available(1) is False
    assert runtime_overlays.required_diffusers_provider_available(BNB_ID) is False


def test_candidate_catalog_or_environment_cannot_grant_provider_authority(monkeypatch):
    _bare, _composite, _selected = _qualified_context(monkeypatch)
    monkeypatch.setenv("MODIFF_RUNTIME_OVERLAY_STATUS", "active")
    monkeypatch.setenv("MODIFF_DIFFUSERS_EVALUATION_PROFILE", "untrusted-selector")
    monkeypatch.setattr(runtime_overlays, "active_diffusers_provider", lambda: None)
    assert runtime_overlays.required_diffusers_provider_available(PROVIDER_ID) is False


def test_composite_exact_declaration_preserves_conflict_and_base_ownership_guards():
    composite = _profile(COMPOSITE_ID)
    packages = [item.to_spec_dict() for item in composite.packages]
    record = {"kind": "optional_runtime", "id": composite.id,
              "specDigest": composite.spec_digest, "spec": composite.to_spec_dict()}
    selected = runtime_overlays.declared_diffusers_provider(packages, [record])
    assert selected["profileId"] == composite.id
    assert selected["specDigest"] == composite.spec_digest
    with pytest.raises(RuntimeError, match="exactly one"):
        runtime_overlays.declared_diffusers_provider(packages, [record, record])
    for name in ("torch", "torchvision", "transformers", "peft"):
        with pytest.raises(RuntimeError, match="other base"):
            runtime_overlays.declared_diffusers_provider([*packages, {"distribution": name}], [record])


def test_existing_catalog_satisfaction_requires_current_declared_composite_digest():
    from modiff import optimization_packages

    composite = _profile(COMPOSITE_ID)
    record = {"kind": "optional_runtime", "id": composite.id, "specDigest": composite.spec_digest}
    for profile_id in (PROVIDER_ID, BNB_ID):
        expected = {"id": profile_id, "specDigest": _profile(profile_id).spec_digest}
        assert optimization_packages._environment_spec_satisfies_optional_profile(record, expected) is True
        stale = {**record, "specDigest": "sha256:" + "0" * 64}
        assert optimization_packages._environment_spec_satisfies_optional_profile(stale, expected) is False
        wrong_expected = {**expected, "specDigest": "sha256:" + "0" * 64}
        assert optimization_packages._environment_spec_satisfies_optional_profile(record, wrong_expected) is False
