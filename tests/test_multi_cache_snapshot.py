"""Execution resolves the same installed roots as model discovery, without downloads."""
import logging
import json

import pytest
from huggingface_hub.errors import CorruptedCacheException

from utils import huggingface as hf

REVISION = "a" * 40


@pytest.fixture
def selected_cache(tmp_path, monkeypatch):
    from modiff import model_artifact_catalog as catalog
    # Load existing profile declarations before narrowing this test's catalog.
    from modiff.auto_resource import artifact_cache_status  # noqa: F401
    primary, secondary = tmp_path / "primary", tmp_path / "secondary"
    primary.mkdir()
    repo = "example/selected"
    snapshot = secondary / "models--example--selected" / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    files = {"modular_model_index.json": b"{}", "variants/fp8r/model.safetensors": b"weights"}
    for name, body in files.items():
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    plan = {"revision": REVISION, "files": [
        {"name": "modular_model_index.json", "size": 2},
        {"name": "native.safetensors", "size": 6},
    ]}
    (snapshot / "native.safetensors").write_bytes(b"native")
    receipt = snapshot.parent.parent / ".modiff_download_plan.json"
    receipt.write_text(json.dumps(plan))
    inventories = [{"repo": repo, "revision": REVISION, "files": [
        {"path": name, "byteSize": len(body)} for name, body in files.items()
    ]}]
    monkeypatch.setattr(catalog, "read_model_artifact_catalog", lambda: {"models": [], "downloadInventories": inventories})
    monkeypatch.setattr(hf, "_hf_cache_locations", lambda: [("configured", str(primary)), ("secondary", str(secondary))])
    monkeypatch.setattr(hf, "HfApi", lambda *args, **kwargs: pytest.fail("Cache observation must not call the Hub"))
    record = {"id": repo, "cache_dirs": [str(primary), str(secondary)], "revisions": [{"hash": REVISION}]}
    return primary, snapshot, receipt, record, inventories


@pytest.mark.parametrize("native_complete", [True, False])
def test_observed_selection_ignores_old_native_receipt(selected_cache, native_complete):
    from modiff.auto_resource import artifact_cache_status
    primary, snapshot, receipt, record, _ = selected_cache
    before = receipt.read_bytes()
    if not native_complete:
        (snapshot / "native.safetensors").unlink()
    status = artifact_cache_status(record["id"], [record])
    assert status.get("observedSelections") == [{
        "revision": REVISION,
        "files": ["modular_model_index.json", "variants/fp8r/model.safetensors"],
        "complete": True,
    }]
    assert status["complete"] is native_complete
    assert status["plannedFiles"] == ["modular_model_index.json", "native.safetensors"]
    assert receipt.read_bytes() == before
    assert list(primary.iterdir()) == []


def test_observed_variants_remain_separate_without_a_receipt_or_weight_reads(selected_cache, monkeypatch):
    from modiff.auto_resource import artifact_cache_status
    _, snapshot, receipt, record, inventories = selected_cache
    receipt.unlink()
    inventories.append({"repo": record["id"], "revision": REVISION, "files": [
        {"path": "modular_model_index.json", "byteSize": 2},
        {"path": "variants/nvfp4/model.safetensors", "byteSize": 4},
    ]})
    original_open = hf.Path.open

    def open_metadata_only(path, *args, **kwargs):
        assert path.suffix != ".safetensors", "Discovery must not read or hash model payloads"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(hf.Path, "open", open_metadata_only)
    status = artifact_cache_status(record["id"], [record])
    assert [item["complete"] for item in status["observedSelections"]] == [True, False]
    assert not receipt.exists()
    assert (snapshot / "variants/fp8r/model.safetensors").stat().st_size == 7


@pytest.mark.parametrize("problem", ["missing", "size", "revision", "escape", "invalid_path", "split_roots", "earlier_incomplete"])
def test_observed_selection_rejects_incomplete_or_different_loader_snapshot(selected_cache, tmp_path, problem):
    from modiff.auto_resource import artifact_cache_status
    primary, snapshot, _, record, inventories = selected_cache
    weight = snapshot / "variants/fp8r/model.safetensors"
    if problem == "missing":
        weight.unlink()
    elif problem == "size":
        weight.write_bytes(b"bad")
    elif problem == "revision":
        inventories[0]["revision"] = "b" * 40
    elif problem == "escape":
        outside = tmp_path / "outside.safetensors"
        outside.write_bytes(b"weights")
        weight.unlink()
        weight.symlink_to(outside)
    elif problem == "invalid_path":
        inventories[0]["files"].append({"path": "../outside.safetensors", "byteSize": 7})
    else:
        first = primary / snapshot.relative_to(snapshot.parents[2])
        first.mkdir(parents=True)
        (first / "modular_model_index.json").write_bytes(b"{}")
        if problem == "split_roots":
            (snapshot / "modular_model_index.json").unlink()
        assert hf.exact_cached_snapshot_path(record["id"], REVISION, "modular_model_index.json") == first
    status = artifact_cache_status(record["id"], [record])
    observations = status.get("observedSelections")
    assert observations and observations[0]["complete"] is False


def test_startup_cache_override_keeps_the_user_default_discoverable(tmp_path, monkeypatch):
    monkeypatch.setitem(hf.CONFIG.hf, "cache_dir", str(tmp_path / "configured"))
    monkeypatch.setattr(hf, "HUGGINGFACE_HUB_CACHE", str(tmp_path / "configured"))
    monkeypatch.setattr(hf.Path, "home", classmethod(lambda cls: tmp_path / "user"))
    monkeypatch.setattr(hf, "_common_appdata_hf_cache_candidates", lambda: [])
    monkeypatch.setattr(hf, "_discovered_appdata_hf_cache_roots", lambda: [])
    assert str(tmp_path / "user/.cache/huggingface/hub") in [path for _, path in hf._hf_cache_locations()]


@pytest.fixture
def caches(tmp_path, monkeypatch):
    primary, secondary = tmp_path / "primary", tmp_path / "secondary"
    primary.mkdir()
    snapshot = secondary / "models--example--audio" / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text("{}")
    monkeypatch.setitem(hf.CONFIG.hf, "cache_dir", str(primary))
    monkeypatch.setattr(hf, "_hf_cache_locations", lambda: [("configured", str(primary)), ("default", str(secondary))])
    return primary, secondary, snapshot


def test_exact_installed_secondary_snapshot_is_loadable(caches):
    _, _, snapshot = caches
    assert hf.exact_cached_snapshot_path("example/audio", REVISION) == snapshot


@pytest.mark.parametrize("canonical_alias", [False, True])
def test_cache_root_alias_accepts_lexical_and_canonical_files(tmp_path, monkeypatch, canonical_alias):
    actual = tmp_path / "actual"
    actual.mkdir()
    configured = tmp_path / "configured"
    configured.symlink_to(actual, target_is_directory=True)
    weight = actual / "weight.safetensors"
    weight.write_bytes(b"cached")
    monkeypatch.setattr(hf, "_hf_cache_locations", lambda: [("configured", str(configured))])
    alias = weight if canonical_alias else configured / weight.name
    assert hf.resolve_managed_hf_cache_file(alias) == weight.resolve()

    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"private")
    (actual / "escape.safetensors").symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        hf.resolve_managed_hf_cache_file(alias.parent / "escape.safetensors")


def test_primary_cache_wins_and_wrong_revision_never_substitutes(caches):
    primary, _, _ = caches
    snapshot = primary / "models--example--audio" / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text("{}")
    assert hf.exact_cached_snapshot_path("example/audio", REVISION) == snapshot
    with pytest.raises(FileNotFoundError):
        hf.exact_cached_snapshot_path("example/audio", "b" * 40)


def test_cross_cache_symlink_does_not_expand_file_authority(caches, tmp_path):
    primary, secondary, snapshot = caches
    outside = tmp_path / "private.json"
    outside.write_text("{}")
    link = snapshot / "escape.json"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        hf.resolve_managed_hf_cache_file(link)
    other_cache_file = primary / "other.json"
    other_cache_file.write_text("{}")
    linked = secondary / "other.json"
    linked.symlink_to(other_cache_file)
    with pytest.raises(ValueError, match="outside"):
        hf.resolve_managed_hf_cache_file(linked)


def _installed_model(cache, repo_id="example/image"):
    snapshot = cache / ("models--" + repo_id.replace("/", "--")) / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text('{"_class_name": "QwenImagePipeline"}')
    return snapshot


@pytest.mark.parametrize("compact", [False, True])
def test_discovery_skips_absent_secondary_cache_without_error(tmp_path, monkeypatch, caplog, compact):
    primary, secondary = tmp_path / "configured", tmp_path / "unused-default"
    _installed_model(primary)
    monkeypatch.setattr(hf, "_hf_cache_locations", lambda: [("configured", str(primary)), ("default", str(secondary))])
    monkeypatch.setattr(hf.logger, "propagate", True)

    with caplog.at_level(logging.DEBUG, logger="modiff"):
        models = hf.get_local_models(compact=compact)

    assert [model["id"] for model in models] == ["example/image"]
    assert models[0]["class_names"] == ["QwenImagePipeline"]
    if not compact:
        assert models[0]["cache_dir"] == str(primary)
        assert models[0]["cache_dirs"] == [str(primary)]
    assert not secondary.exists()
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)
    assert any(record.levelno == logging.DEBUG and str(secondary) in record.message for record in caplog.records)


def test_discovery_still_scans_existing_secondary_cache(tmp_path, monkeypatch):
    primary, secondary = tmp_path / "configured", tmp_path / "default"
    _installed_model(primary, "example/primary")
    _installed_model(secondary, "example/secondary")
    monkeypatch.setattr(hf, "_hf_cache_locations", lambda: [("configured", str(primary)), ("default", str(secondary))])

    models = hf.get_local_models()

    assert [model["id"] for model in models] == ["example/primary", "example/secondary"]
    assert [model["cache_dir"] for model in models] == [str(primary), str(secondary)]


@pytest.mark.parametrize("failure", ["stat_permission", "scan_permission", "scan_corruption"])
def test_discovery_retains_real_cache_errors_and_other_locations(tmp_path, monkeypatch, caplog, failure):
    primary, secondary = tmp_path / "configured", tmp_path / "default"
    primary.mkdir()
    _installed_model(secondary)
    monkeypatch.setattr(hf, "_hf_cache_locations", lambda: [("configured", str(primary)), ("default", str(secondary))])
    monkeypatch.setattr(hf.logger, "propagate", True)
    real_scan = hf.scan_cache_dir
    real_stat = hf.Path.stat
    error = CorruptedCacheException("corrupt cache") if failure == "scan_corruption" else PermissionError("cache access denied")

    def scan(cache_dir):
        if cache_dir == str(primary):
            raise error
        return real_scan(cache_dir)

    def stat(path, *args, **kwargs):
        if path == primary:
            raise error
        return real_stat(path, *args, **kwargs)

    if failure == "stat_permission":
        monkeypatch.setattr(hf.Path, "stat", stat)
    else:
        monkeypatch.setattr(hf, "scan_cache_dir", scan)
    with caplog.at_level(logging.DEBUG, logger="modiff"):
        models = hf.get_local_models()

    assert [model["id"] for model in models] == ["example/image"]
    assert models[0]["cache_dir"] == str(secondary)
    errors = [record.message for record in caplog.records if record.levelno >= logging.ERROR]
    assert errors == [f"Error scanning cache directory {primary}: {error}"]


def test_discovery_retains_non_directory_cache_error(tmp_path, monkeypatch, caplog):
    primary, secondary = tmp_path / "configured", tmp_path / "default"
    primary.write_text("wrong cache path")
    _installed_model(secondary)
    monkeypatch.setattr(hf, "_hf_cache_locations", lambda: [("configured", str(primary)), ("default", str(secondary))])
    monkeypatch.setattr(hf.logger, "propagate", True)

    with caplog.at_level(logging.DEBUG, logger="modiff"):
        models = hf.get_local_models()

    assert [model["id"] for model in models] == ["example/image"]
    errors = [record.message for record in caplog.records if record.levelno >= logging.ERROR]
    assert len(errors) == 1 and str(primary) in errors[0] and "expects a directory" in errors[0]
