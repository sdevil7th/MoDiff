"""Auxiliary overlays must use the application's verified native model libraries."""

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from modiff import optional_runtimes as runtimes


EXPECTED_STAGED = {
    runtimes.TRANSFORMERS_MAIN_PEFT_QUANTO_RUNTIME_PROFILE_ID: {"optimum-quanto", "ninja"},
    runtimes.TRANSFORMERS_MAIN_PEFT_GGUF_RUNTIME_PROFILE_ID: {"gguf"},
    runtimes.TRANSFORMERS_MAIN_PEFT_BITSANDBYTES_RUNTIME_PROFILE_ID: {"bitsandbytes"},
    runtimes.GALLERY_MEDIA_RUNTIME_PROFILE_ID: {"opencv-python-headless", "av"},
}
PREVIOUS_COMPOSITE_DIGESTS = {
    runtimes.TRANSFORMERS_MAIN_PEFT_QUANTO_RUNTIME_PROFILE_ID:
        "sha256:d17897a8e4a58d4ad2b891527772b5853dc8b91abdbfaf9639480ae5a5840f04",
    runtimes.TRANSFORMERS_MAIN_PEFT_GGUF_RUNTIME_PROFILE_ID:
        "sha256:11154ca9ff598bdadf6bcda3c029c7a04c59b104e589ffc16efbef356cda22fb",
    runtimes.TRANSFORMERS_MAIN_PEFT_BITSANDBYTES_RUNTIME_PROFILE_ID:
        "sha256:4bab84d72fc81e613c3df9d753906bb4739f484531a8744a37259000be8cfce9",
    runtimes.GALLERY_MEDIA_RUNTIME_PROFILE_ID:
        "sha256:aa94349bfaa3f0cbd448897c07b1a89f6276430ee71377b0f7d59005732f79ea",
}


@pytest.mark.parametrize("profile_id,staged", EXPECTED_STAGED.items())
def test_auxiliary_profiles_stage_only_their_immutable_extra_packages(profile_id, staged):
    profile = runtimes.OPTIONAL_RUNTIME_PROFILES[profile_id]
    assert {package.distribution for package in profile.packages} == staged
    assert {lock["distribution"] for lock in profile.artifact_locks} == staged
    assert not profile.source_builds and not profile.satisfies_profiles
    assert "Transformers" not in profile.label and "PEFT" not in profile.label
    base = {package.distribution: package for package in profile.base_packages}
    assert base["transformers"].specifier == ">=5.18.0"
    assert base["peft"].specifier == ">=0.21.2"
    assert base["huggingface-hub"].specifier == ">=1.31.0,<2.0"
    assert not staged.intersection(base)
    assert {"torch", "tokenizers", "typer", "safetensors", "numpy", "requests"} <= set(base)
    assert profile.spec_digest != PREVIOUS_COMPOSITE_DIGESTS[profile_id]


@pytest.mark.parametrize("profile_id", EXPECTED_STAGED)
def test_auxiliary_base_floors_follow_required_project_dependencies(profile_id):
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]
    requirements = {item.name: item for item in map(Requirement, project["dependencies"])}
    base = {package.distribution: package for package in runtimes.OPTIONAL_RUNTIME_PROFILES[profile_id].base_packages}
    for name in ("transformers", "peft", "huggingface-hub", "accelerate"):
        assert str(Requirement(name + base[name].specifier).specifier) == str(requirements[name].specifier)


@pytest.mark.parametrize("profile_id", EXPECTED_STAGED)
def test_auxiliary_actions_remain_scoped_to_checked_installation_target(profile_id):
    profile = runtimes.OPTIONAL_RUNTIME_PROFILES[profile_id]
    for platform_name, _, machine in runtimes._OPTIONAL_RUNTIME_TARGETS:
        policy = profile.contract_for_target(platform_name=platform_name, machine=machine)
        qualified = (
            (platform_name, machine) == ("linux", "x86_64")
            and profile_id != runtimes.TRANSFORMERS_MAIN_PEFT_BITSANDBYTES_RUNTIME_PROFILE_ID
        )
        assert (policy.cutover_ready, policy.install_action_available, policy.activation_available) == (qualified,) * 3
        assert policy.contract_state == ("qualified" if qualified else "candidate_unqualified")
    assert not profile.cutover_ready and not profile.install_action_available and not profile.activation_available


def test_bitsandbytes_new_native_base_needs_accelerator_qualification():
    profile = runtimes.OPTIONAL_RUNTIME_PROFILES[runtimes.TRANSFORMERS_MAIN_PEFT_BITSANDBYTES_RUNTIME_PROFILE_ID]
    for platform_name, _, machine in runtimes._OPTIONAL_RUNTIME_TARGETS:
        policy = profile.contract_for_target(platform_name=platform_name, machine=machine)
        assert not policy.install_action_available and not policy.activation_available
        assert not policy.cutover_ready


def test_auxiliary_wheel_bytes_keep_the_existing_reviewed_identity():
    gguf = runtimes.OPTIONAL_RUNTIME_PROFILES[runtimes.TRANSFORMERS_MAIN_PEFT_GGUF_RUNTIME_PROFILE_ID]
    assert gguf.artifact_locks == runtimes._gguf_artifact_locks()
    quanto = runtimes.OPTIONAL_RUNTIME_PROFILES[runtimes.TRANSFORMERS_MAIN_PEFT_QUANTO_RUNTIME_PROFILE_ID]
    assert quanto.artifact_locks == runtimes._quanto_linux_x86_64_artifact_locks()
    bitsandbytes = runtimes.OPTIONAL_RUNTIME_PROFILES[runtimes.TRANSFORMERS_MAIN_PEFT_BITSANDBYTES_RUNTIME_PROFILE_ID]
    assert bitsandbytes.artifact_locks == runtimes._bitsandbytes_linux_x86_64_artifact_locks()
