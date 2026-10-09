"""Native installation readiness, required model libraries and safe migration."""

from importlib import metadata
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import tomllib
from unittest.mock import Mock

import pytest

from modiff import base_runtime, optimization_packages, runtime_profile, runtime_overlays
from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
from modiff.optional_runtime_execution import graph_optional_runtime_requirement
from modiff.optional_runtimes import GALLERY_MEDIA_RUNTIME_PROFILE_ID, TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "fbf49e7f35857f76bc57b177e26f12b03687c668"


@pytest.fixture
def required_packages(tmp_path, monkeypatch):
    project = tmp_path / "pyproject.toml"
    project.write_text(f'''[project]
requires-python = ">=3.12,<3.13"
dependencies = ["torch>=2.6.0", "transformers>=5.18.0", "peft>=0.21.2",
"diffusers @ git+https://github.com/huggingface/diffusers.git@{COMMIT}"]
''')
    monkeypatch.setattr(base_runtime, "PROJECT_METADATA", project)
    versions = {"torch": "2.14.1+cpu", "transformers": "5.18.0", "peft": "0.21.2", "diffusers": "0.41.0.dev0"}
    source = {"url": "https://github.com/huggingface/diffusers.git", "vcs_info": {"vcs": "git", "commit_id": COMMIT}}

    def version(name):
        if name not in versions:
            raise metadata.PackageNotFoundError(name)
        return versions[name]

    def distribution(_name):
        return SimpleNamespace(read_text=lambda _: json.dumps(source))

    return versions, source, lambda: base_runtime.base_runtime_status(version_resolver=version, distribution_resolver=distribution)


def test_native_base_requires_transformers_and_peft_on_every_platform():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    for name in ("transformers", "peft"):
        declaration = next(item for item in project["project"]["dependencies"] if item.startswith(name))
        assert ";" not in declaration and "==" not in declaration
    assert project["tool"]["uv"].get("managed") is not False
    assert (ROOT / "uv.lock").is_file()
    assert "cuda-optimizations" not in project["project"]["optional-dependencies"]
    for accelerator, index in (("cpu", "pytorch-cpu"), ("cuda", "pytorch-cu128"), ("xpu", "pytorch-xpu")):
        assert any(source.get("extra") == accelerator and source["index"] == index for source in project["tool"]["uv"]["sources"]["torch"])


def test_native_readiness_checks_versions_and_reviewed_source_without_importing_models(required_packages, monkeypatch):
    versions, source, observe = required_packages
    original_import = __import__

    def safe_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "peft", "diffusers"}:
            pytest.fail(f"Readiness must not import {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", safe_import)
    assert observe()["verified"]
    versions.pop("peft")
    assert observe()["issues"] == ["Missing required package: peft."]
    versions["peft"] = "0.20.0"
    assert not observe()["verified"]
    versions["peft"] = "0.21.2"
    source["vcs_info"]["commit_id"] = "0" * 40
    assert not observe()["verified"]


@pytest.mark.parametrize("direct", [
    [], {"vcs_info": []},
    {"url": "https://github.com/example/diffusers.git", "vcs_info": {"vcs": "git", "commit_id": COMMIT}},
    {"url": "https://github.com/huggingface/diffusers.git", "vcs_info": {"vcs": "hg", "commit_id": COMMIT}},
])
def test_native_readiness_rejects_malformed_or_wrong_repository_provenance(required_packages, direct):
    versions, _source, _observe = required_packages
    status = base_runtime.base_runtime_status(
        version_resolver=versions.__getitem__,
        distribution_resolver=lambda _name: SimpleNamespace(read_text=lambda _file: json.dumps(direct)),
    )
    assert not status["verified"]
    assert status["issues"] == [f"diffusers must use the declared source revision {COMMIT}."]


def test_native_readiness_accepts_canonical_git_url_with_or_without_dot_git(required_packages):
    _versions, source, observe = required_packages
    source["url"] = source["url"].removesuffix(".git")
    assert observe()["verified"]


def verified_base():
    return {"status": "verified", "verified": True, "matches": True, "issues": [],
            "installationMode": "uv", "current_digest": "a" * 64, "packages": {"torch": "2.14.1+cpu"}}


def test_native_runtime_needs_no_managed_installation_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_profile, "base_runtime_status", verified_base)
    monkeypatch.setattr(runtime_profile, "_device_tensor_probe", lambda *args: {"ready": True})
    observed = runtime_profile.runtime_profile({"torch": {"available": True, "version": "2.14.1+cpu"}}, venv=tmp_path)
    assert observed["execution_ready"] and observed["runtime_contract"]["verified"]
    assert not any(issue["code"] == "profile-unverified" for issue in observed["issues"])


def test_valid_legacy_receipt_cannot_hide_missing_required_base_packages(tmp_path, monkeypatch):
    (tmp_path / "modiff-profile.json").write_text(json.dumps({"profile": "cpu", "lock_digest": "a" * 64}))
    monkeypatch.setattr(runtime_profile, "base_runtime_status", lambda: {**verified_base(), "verified": False,
        "status": "incompatible", "issues": ["Missing required package: peft."]})
    monkeypatch.setattr(runtime_profile, "_device_tensor_probe", lambda *args: {"ready": True})
    observed = runtime_profile.runtime_profile({"torch": {"available": True}}, venv=tmp_path)
    assert not observed["execution_ready"] and observed["repair_required"]
    assert observed["repair_command"] == "uv sync --extra cpu"


@pytest.mark.parametrize("os_name,architecture,execution_ready", [
    ("linux", "x86_64", True), ("windows", "x86_64", True), ("macos", "arm64", False),
])
def test_native_accelerator_selection_supersedes_a_stale_cpu_receipt(
    tmp_path, monkeypatch, os_name, architecture, execution_ready,
):
    (tmp_path / "modiff-profile.json").write_text(json.dumps({"profile": "cpu"}))
    monkeypatch.setattr(runtime_profile, "normalized_os", lambda: os_name)
    monkeypatch.setattr(runtime_profile, "normalized_arch", lambda: architecture)
    monkeypatch.setattr(runtime_profile, "base_runtime_status", lambda: {
        **verified_base(), "packages": {"torch": "2.11.0+cu128"},
    })
    monkeypatch.setattr(runtime_profile, "_device_tensor_probe", lambda *args: {"ready": True})
    observed = runtime_profile.runtime_profile({"torch": {"available": True, "cuda_available": True,
        "version": "2.11.0+cu128", "cuda_version": "12.8"}}, venv=tmp_path)
    assert observed["installed"] == "nvidia-cuda"
    assert observed["execution_ready"] is execution_ready
    assert ("unsupported-platform" in {issue["code"] for issue in observed["issues"]}) is (not execution_ready)


def test_normal_text_to_image_has_no_optional_install_or_activation_gate(monkeypatch):
    assert DIFFUSERS_EXECUTION_PROFILES["z-image:auto"].optional_runtime_profiles == ()
    monkeypatch.setenv("MODIFF_RUNTIME_OVERLAY_STATUS", "repair_required")
    graph = {"nodes": {"loader": {"module": "modules.DiffusersImage", "action": "LoadPipeline", "params": {
        "pipeline_class": {"value": "ZImagePipeline"},
        "model_id": {"value": {"source": "hub", "value": "Tongyi-MAI/Z-Image-Turbo"}},
    }}}, "paths": [["loader"]]}
    observe_overlay = Mock(side_effect=AssertionError("Core workflow must not inspect optional environments"))
    requirement = graph_optional_runtime_requirement(graph, catalog_resolver=observe_overlay)
    assert requirement["state"] == "base_satisfied" and not requirement["requiredNow"]
    observe_overlay.assert_not_called()


@pytest.mark.parametrize("profile_id,expected_status", [(TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID, "base"), (GALLERY_MEDIA_RUNTIME_PROFILE_ID, "repair_required")])
def test_obsolete_core_overlays_never_shadow_native_libraries(monkeypatch, profile_id, expected_status):
    state = {"activeEnvironmentId": "runtime-1-12345678", "activeTrustClass": "artifact_locked_optional", "_storageStatus": "ok"}
    saved = Mock()
    monkeypatch.setattr(base_runtime, "base_runtime_status", verified_base)
    monkeypatch.setattr(optimization_packages, "reserve_install", lambda *args: object())
    monkeypatch.setattr(optimization_packages, "release_install", lambda *args: None)
    monkeypatch.setattr(optimization_packages, "_reconcile_promotion", lambda *args: None)
    monkeypatch.setattr(optimization_packages, "read_state", lambda: state.copy())
    monkeypatch.setattr(optimization_packages, "_environment_inspection", lambda *args, **kwargs: {
        "status": "recorded", "manifest": {"specs": [{"kind": "optional_runtime", "id": profile_id}],
        "packageContracts": [{"distribution": "transformers"}, {"distribution": "peft"}]}})
    monkeypatch.setattr(optimization_packages, "_write_state", saved)
    monkeypatch.setattr(optimization_packages, "_safe_environment_path", lambda *args, **kw: pytest.fail("Never import obsolete overlay bytes"))
    before = list(sys.path)
    assert optimization_packages.activate_runtime_overlay() is None
    assert sys.path == before and os.environ["MODIFF_RUNTIME_OVERLAY_STATUS"] == expected_status
    if expected_status == "base":
        migrated = saved.call_args.args[0]
        assert migrated["activeEnvironmentId"] is None and migrated["previousEnvironmentId"] == state["activeEnvironmentId"]
    else:
        saved.assert_not_called()


@pytest.mark.parametrize("platform_name,profile_id", [
    ("linux", "cpu"), ("win32", "cpu"), ("darwin", "apple-mps"),
])
def test_native_optional_binding_uses_observed_dependencies_instead_of_receipts(
    tmp_path, monkeypatch, platform_name, profile_id,
):
    version = tmp_path / "version.py"
    version.write_text("cuda = None\nhip = None\n")
    monkeypatch.setattr(runtime_overlays.sys, "platform", platform_name)
    monkeypatch.setattr(runtime_overlays.metadata, "distribution", lambda _: SimpleNamespace(locate_file=lambda _: version))
    monkeypatch.setattr(base_runtime, "base_runtime_status", verified_base)
    monkeypatch.setattr(runtime_profile, "read_state", lambda *args: None)
    identity = runtime_overlays._accelerator_identity()
    assert identity["profileId"] == profile_id and identity["lockDigest"] == "a" * 64
    assert {Path(item["path"]).name for item in identity["contractFiles"]} == {"pyproject.toml", "uv.lock", "accelerators.v1.json"}


def test_native_optional_binding_detects_pypi_cuda_builds_without_a_version_suffix(tmp_path, monkeypatch):
    version = tmp_path / "version.py"
    version.write_text("cuda: str = '13.0'\nhip = None\n")
    monkeypatch.setattr(base_runtime, "base_runtime_status", lambda: {**verified_base(), "packages": {"torch": "2.11.0"}})
    monkeypatch.setattr(runtime_profile, "read_state", lambda *args: None)
    monkeypatch.setattr(runtime_overlays.metadata, "distribution", lambda _: SimpleNamespace(locate_file=lambda _: version))
    assert runtime_overlays._accelerator_identity()["profileId"] == "nvidia-cuda"


def test_reset_to_base_clears_only_selection_pointers_under_the_install_lease(monkeypatch):
    state = {"activeEnvironmentId": "runtime-1-12345678", "previousEnvironmentId": "runtime-2-12345678",
        "activeTrustClass": "artifact_locked_optional", "previousTrustClass": "artifact_locked_optional",
        "enabledCapabilities": ["torchao"], "_storageStatus": "ok"}
    reserved = Mock(return_value=object())
    released = Mock()
    saved = Mock(side_effect=lambda value: value)
    monkeypatch.setattr(base_runtime, "base_runtime_status", verified_base)
    monkeypatch.setattr(optimization_packages, "reserve_install", reserved)
    monkeypatch.setattr(optimization_packages, "release_install", released)
    monkeypatch.setattr(optimization_packages, "_reconcile_promotion", lambda *args: None)
    monkeypatch.setattr(optimization_packages, "read_state", lambda: state.copy())
    monkeypatch.setattr(optimization_packages, "_write_state", saved)
    monkeypatch.setattr(optimization_packages, "remove_managed_directory", lambda *args, **kw: pytest.fail("Reset must preserve artifacts"))
    with pytest.raises(ValueError, match="consent"):
        optimization_packages.reset_optional_runtime_to_base(consent=False)
    reserved.assert_not_called()
    result = optimization_packages.reset_optional_runtime_to_base(consent=True)
    assert result["restartRequired"] and result["target"] == "base"
    assert result["state"]["enabledCapabilities"] == ["torchao"]
    assert all(result["state"][key] is None for key in ("activeEnvironmentId", "previousEnvironmentId", "activeTrustClass", "previousTrustClass"))
    released.assert_called_once_with(reserved.return_value)


def test_reset_to_base_requires_a_verified_native_base_before_mutation(monkeypatch):
    monkeypatch.setattr(base_runtime, "base_runtime_status", lambda: {"verified": False})
    reserved = Mock()
    monkeypatch.setattr(optimization_packages, "reserve_install", reserved)
    with pytest.raises(RuntimeError, match="Repair the required base"):
        optimization_packages.reset_optional_runtime_to_base(consent=True)
    reserved.assert_not_called()
