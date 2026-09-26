"""Execution resolves the same installed roots as model discovery, without downloads."""
import pytest

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
