"""Execution resolves the same installed roots as model discovery, without downloads."""
import logging

import pytest
from huggingface_hub.errors import CorruptedCacheException

from utils import huggingface as hf

REVISION = "a" * 40


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
