"""Damaged temporary distribution metadata must yield actionable base readiness."""
import builtins
from importlib import metadata

import pytest

from modiff import base_runtime, diffusers_evaluation_runtime, runtime_overlays, runtime_profile


@pytest.fixture
def temporary_base(tmp_path, monkeypatch):
    project = tmp_path / "pyproject.toml"
    project.write_text('''[project]
requires-python = ">=3.12,<3.13"
dependencies = ["peft>=0.21.2", "transformers>=5.18.0", "diffusers==0.41.0"]
''', encoding="utf-8")
    monkeypatch.setattr(base_runtime, "PROJECT_METADATA", project)
    monkeypatch.setattr(runtime_overlays, "active_diffusers_provider", lambda: None)
    monkeypatch.setattr(diffusers_evaluation_runtime, "selected_diffusers_evaluation_contract", lambda **_: None)
    distributions, metadata_files = {}, {}
    for name, version in [("peft", "0.21.2"), ("transformers", "5.18.0"), ("diffusers", "0.41.0")]:
        directory = tmp_path / f"{name}-{version}.dist-info"
        directory.mkdir()
        (directory / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8")
        distributions[name] = metadata.Distribution.at(directory)
        metadata_files[name] = directory / "METADATA"
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "transformers", "peft", "diffusers"}
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    def observe():
        return base_runtime.base_runtime_status(
            version_resolver=lambda name: distributions[name].version,
            distribution_resolver=distributions.__getitem__,
        )
    return distributions, metadata_files, observe


@pytest.mark.parametrize("version", [None, "", "not-a-version", "broken\n", " "])
@pytest.mark.parametrize("package", ["peft", "transformers", "diffusers"])
def test_real_damaged_metadata_is_unverified_without_import_or_repair(temporary_base, package, version):
    _distributions, metadata_files, observe = temporary_base
    path = metadata_files[package]
    version_line = "" if version is None else f"Version: {version}\n"
    path.write_text(f"Metadata-Version: 2.1\nName: {package}\n{version_line}", encoding="utf-8")
    before = path.read_bytes()
    status = observe()
    assert status["status"] == "incompatible" and not status["verified"] and not status["matches"]
    assert status["issues"] == [f"Invalid required package version metadata: {package}."]
    assert package not in status["packages"]
    assert all(isinstance(value, str) for value in status["packages"].values())
    assert path.read_bytes() == before


@pytest.mark.parametrize("error", [PermissionError("private path"), UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")])
def test_unreadable_metadata_is_reported_without_exposing_paths(temporary_base, error):
    distributions, _metadata_files, _ = temporary_base
    def version(name):
        if name == "peft":
            raise error
        return distributions[name].version
    status = base_runtime.base_runtime_status(version_resolver=version, distribution_resolver=distributions.__getitem__)
    assert not status["verified"] and status["issues"] == ["Cannot read required package metadata: peft."]
    assert "peft" not in status["packages"]


def test_valid_metadata_and_versions_outside_the_requirement_remain_distinct(temporary_base):
    _distributions, metadata_files, observe = temporary_base
    assert observe()["verified"]
    path = metadata_files["peft"]
    path.write_text("Name: peft\nVersion: 0.1.0\n", encoding="utf-8")
    status = observe()
    assert not status["verified"] and status["packages"]["peft"] == "0.1.0"
    assert status["issues"] == ["peft 0.1.0 does not satisfy >=0.21.2."]


@pytest.mark.parametrize("version", ["v0.21.2", "0.21.2+local"])
def test_valid_pep440_spelling_is_preserved(temporary_base, version):
    _distributions, metadata_files, observe = temporary_base
    metadata_files["peft"].write_text(f"Name: peft\nVersion: {version}\n", encoding="utf-8")
    status = observe()
    assert status["verified"] and status["packages"]["peft"] == version


def test_unrelated_resolver_errors_are_not_silenced(temporary_base):
    distributions, _metadata_files, _observe = temporary_base
    def version(_name):
        raise RuntimeError("unrelated resolver defect")
    with pytest.raises(RuntimeError, match="unrelated resolver defect"):
        base_runtime.base_runtime_status(version_resolver=version, distribution_resolver=distributions.__getitem__)


@pytest.mark.parametrize("version", [None, "invalid-version"])
def test_provider_cannot_hide_invalid_underlying_base_version(temporary_base, monkeypatch, version):
    _distributions, metadata_files, observe = temporary_base
    metadata_files["diffusers"].write_text("Name: diffusers\nVersion: 0.42.0.dev0\n", encoding="utf-8")
    monkeypatch.setattr(runtime_overlays, "active_diffusers_provider", lambda: {
        "profileId": "test-provider", "specDigest": "a" * 64, "distribution": "diffusers",
        "version": "0.42.0.dev0", "recordDigest": "b" * 64,
    })
    monkeypatch.setattr(runtime_overlays, "base_diffusers_identity", lambda: {
        "version": version, "declaration": "diffusers==0.41.0", "recordDigest": "c" * 64,
    })
    status = observe()
    assert not status["verified"] and status["issues"] == ["Invalid required package version metadata: diffusers."]


def test_damaged_native_metadata_keeps_existing_repair_guidance(temporary_base, tmp_path, monkeypatch):
    _distributions, metadata_files, observe = temporary_base
    path = metadata_files["peft"]
    path.write_text("Name: peft\n", encoding="utf-8")
    monkeypatch.setattr(runtime_profile, "base_runtime_status", observe)
    monkeypatch.setattr(runtime_profile, "normalized_os", lambda: "linux")
    monkeypatch.setattr(runtime_profile, "normalized_arch", lambda: "x86_64")
    monkeypatch.setattr(runtime_profile, "_device_tensor_probe", lambda *args: {"ready": True})
    result = runtime_profile.runtime_profile({"torch": {"available": True}}, venv=tmp_path)
    assert not result["execution_ready"] and result["repair_required"]
    assert result["repair_command"] == "uv sync --extra cpu"
    assert not result["runtime_contract"]["verified"]
