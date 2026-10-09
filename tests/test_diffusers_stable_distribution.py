"""Ordinary Diffusers readiness and immutable overlay installation identity."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

from modiff import base_runtime, runtime_overlays, install


@pytest.fixture
def ordinary_project(tmp_path, monkeypatch):
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nrequires-python = ">=3.12,<3.13"\ndependencies = ["diffusers>=0.41.0"]\n')
    monkeypatch.setattr(base_runtime, "PROJECT_METADATA", project)
    return project


@pytest.mark.parametrize("version, ready", [("0.41.0", True), ("0.41.1", True), ("0.42.0", True),
    ("0.40.0", False), ("0.41.0.dev0", False)])
def test_ordinary_diffusers_readiness_uses_minimum_without_git_receipt(ordinary_project, version, ready):
    def forbidden(_name):
        pytest.fail("Published-package readiness must not require a Git receipt or model import")
    status = base_runtime.base_runtime_status(version_resolver=lambda _name: version, distribution_resolver=forbidden)
    assert status["verified"] is ready
    assert status["packages"] == {"diffusers": version}


def fake_distribution(tmp_path, *, version="0.41.0", direct=None, record="diffusers/__init__.py,sha256:fixture,1\n"):
    origin = tmp_path / "diffusers-0.41.0.dist-info"
    origin.mkdir(exist_ok=True)
    return SimpleNamespace(version=version, _path=origin,
        read_text=lambda name: json.dumps(direct) if name == "direct_url.json" and direct is not None else
            record if name == "RECORD" else None)


def identity_with(tmp_path, monkeypatch, distribution):
    monkeypatch.setattr(runtime_overlays.metadata, "distribution", lambda name: distribution)
    monkeypatch.setattr(runtime_overlays, "_import_identity", lambda name, origin:
        {"origin": str(tmp_path / "diffusers/__init__.py"), "searchLocations": [str(tmp_path / "diffusers")]})
    return runtime_overlays._diffusers_identity()


def test_overlay_binds_published_diffusers_record_without_git_identity(ordinary_project, tmp_path, monkeypatch):
    distribution = fake_distribution(tmp_path)
    identity = identity_with(tmp_path, monkeypatch, distribution)
    assert identity["version"] == "0.41.0"
    assert identity["declaration"] == "diffusers>=0.41.0"
    assert identity["directUrl"] is None
    assert identity["recordDigest"] == "sha256:" + hashlib.sha256(distribution.read_text("RECORD").encode()).hexdigest()
    assert "commitId" not in identity
    changed = fake_distribution(tmp_path, record="a changed distribution record\n")
    assert identity_with(tmp_path, monkeypatch, changed) != identity


@pytest.mark.parametrize("direct", [[], {"dir_info": {"editable": True}}])
def test_overlay_rejects_malformed_or_editable_diffusers(ordinary_project, tmp_path, monkeypatch, direct):
    with pytest.raises(RuntimeError, match="installed distribution"):
        identity_with(tmp_path, monkeypatch, fake_distribution(tmp_path, direct=direct))


@pytest.mark.parametrize("version", ["0.40.0", "0.41.0.dev0"])
def test_overlay_rejects_below_minimum_and_prerelease(ordinary_project, tmp_path, monkeypatch, version):
    with pytest.raises(RuntimeError, match="does not satisfy"):
        identity_with(tmp_path, monkeypatch, fake_distribution(tmp_path, version=version))


def test_overlay_rejects_absent_distribution_record(ordinary_project, tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="distribution record"):
        identity_with(tmp_path, monkeypatch, fake_distribution(tmp_path, record=None))


def test_fresh_validator_rechecks_record_version_and_direct_source(ordinary_project, tmp_path, monkeypatch):
    # Exercise the exact emitted child-validator section, including its real Requirement parsing.
    script = runtime_overlays._VALIDATION_SCRIPT
    start = script.index('diffusers = metadata.distribution("diffusers")')
    end = script.index('\nassert_import_origin(', start)
    section = script[start:end]
    distribution = fake_distribution(tmp_path)
    binding = identity_with(tmp_path, monkeypatch, distribution)
    import importlib
    namespace = {"metadata": runtime_overlays.metadata, "json": json, "hashlib": hashlib,
                 "Path": Path, "importlib": importlib, "payload": {"binding": {"diffusers": binding}}}
    exec(section, namespace)
    for changed in [fake_distribution(tmp_path, version="0.41.1"),
                    fake_distribution(tmp_path, record="modified\n"),
                    fake_distribution(tmp_path, direct={"url": "https://example.invalid/changed.whl"})]:
        monkeypatch.setattr(runtime_overlays.metadata, "distribution", lambda name, value=changed: value)
        with pytest.raises(RuntimeError, match="identity drifted"):
            exec(section, namespace)


@pytest.mark.parametrize("source, url, digest", [
    ({"git": "https://example.invalid/repository"}, "https://files.pythonhosted.org/diffusers-0.41.0-py3-none-any.whl", "sha256:"+"a"*64),
    ({"registry": "https://pypi.org/simple"}, "https://example.invalid/diffusers-0.41.0-py3-none-any.whl", "sha256:"+"a"*64),
    ({"registry": "https://pypi.org/simple"}, "http://files.pythonhosted.org/diffusers-0.41.0-py3-none-any.whl", "sha256:"+"a"*64),
    ({"registry": "https://pypi.org/simple"}, "https://files.pythonhosted.org/diffusers-0.41.0-py3-none-any.whl", "sha256:not-a-digest"),
])
def test_managed_installer_rejects_wrong_locked_wheel_provenance_before_io(tmp_path, monkeypatch, source, url, digest):
    root = tmp_path / "project"; root.mkdir()
    # tomllib is patched only to supply adversarial lock fields; no installer/network call may occur.
    (root / "uv.lock").write_text("ignored fixture")
    monkeypatch.setattr(install, "ROOT", root)
    monkeypatch.setattr(install.tomllib, "load", lambda _handle: {"package": [{"name": "diffusers", "version": "0.41.0", "source": source,
        "wheels": [{"url": url, "hash": digest}]}]})
    monkeypatch.setattr(install, "_run", lambda *_args, **_kwargs: pytest.fail("Untrusted wheel must be rejected before installation"))
    with pytest.raises(RuntimeError):
        install._install_reviewed_diffusers("uv", tmp_path / "python")


def test_tested_lock_records_official_stable_wheel():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    assert "diffusers>=0.41.0" in project["project"]["dependencies"]
    lock = tomllib.loads((root / "uv.lock").read_text())
    package = next(item for item in lock["package"] if item["name"] == "diffusers")
    assert package["version"] == "0.41.0"
    assert package["source"] == {"registry": "https://pypi.org/simple"}
    assert package["wheels"][0]["hash"] == "sha256:ea8918b7dfd92ce793b6db689a551b8cd25d52c6c0fcd3ade0706a5fd2a25990"


def test_historical_catalog_source_is_separate_from_runtime_minimum(tmp_path):
    from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION
    from modiff.upstream_coverage import _pinned_diffusers_revision, UpstreamCoverageError

    (tmp_path / "data").mkdir()
    snapshot = tmp_path / "data/modular-workflow-contracts.json"
    snapshot.write_text(json.dumps({"diffusersRevision": PINNED_DIFFUSERS_REVISION}))
    (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["diffusers>=0.41.0"]\n')
    assert _pinned_diffusers_revision(tmp_path) == PINNED_DIFFUSERS_REVISION
    snapshot.write_text(json.dumps({"diffusersRevision": "0" * 40}))
    with pytest.raises(UpstreamCoverageError, match="reviewed source revision"):
        _pinned_diffusers_revision(tmp_path)


def test_historical_catalog_source_verification_still_rejects_a_different_revision(tmp_path, monkeypatch):
    from modiff import upstream_coverage
    from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION

    monkeypatch.setattr(upstream_coverage, "_git_diffusers_revision", lambda source: "0" * 40)
    with pytest.raises(upstream_coverage.UpstreamCoverageError, match="expected reviewed pin"):
        upstream_coverage._verify_diffusers_source_revision(tmp_path, PINNED_DIFFUSERS_REVISION)


def test_lumina_artifact_metadata_uses_canonical_runtime_classes():
    import diffusers
    from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS, pipeline_class_from_name

    # Old model_index class names are artifact metadata, not SDK dispatch names.
    first = IMAGE_PIPELINE_ADAPTERS["LuminaPipeline"]
    second = IMAGE_PIPELINE_ADAPTERS["Lumina2Pipeline"]
    assert first.model_filter_classes == ("LuminaText2ImgPipeline",)
    assert first.load_pipeline_class == "LuminaPipeline"
    assert second.load_pipeline_class == "Lumina2Pipeline"
    assert pipeline_class_from_name(first.load_pipeline_class) is diffusers.LuminaPipeline
    assert pipeline_class_from_name(second.load_pipeline_class) is diffusers.Lumina2Pipeline


def test_catalog_override_verifies_exact_source_before_use(tmp_path, monkeypatch):
    from modiff import upstream_coverage
    from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION

    source = tmp_path / "reviewed"
    source.mkdir()
    (source / "__init__.py").write_text("__version__ = 'historical'\n")
    monkeypatch.setenv("MODIFF_DIFFUSERS_CATALOG_SOURCE", str(source))
    monkeypatch.setattr(upstream_coverage, "_git_diffusers_revision", lambda value: PINNED_DIFFUSERS_REVISION)
    assert upstream_coverage.reviewed_diffusers_source() == source
    monkeypatch.setattr(upstream_coverage, "_git_diffusers_revision", lambda value: "0" * 40)
    with pytest.raises(upstream_coverage.UpstreamCoverageError, match="expected reviewed pin"):
        upstream_coverage.reviewed_diffusers_source()
