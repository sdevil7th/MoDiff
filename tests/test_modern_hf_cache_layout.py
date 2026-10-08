"""Exact snapshots support Hub's shared, sharded blob layout without widening authority."""
import hashlib
import json
import os
from pathlib import PurePosixPath

import pytest

from modules.ModularDiffusers.loaders import _read_reviewed_pipeline_index
from utils import huggingface as hf

REVISION = "a" * 40
REPOSITORY = "example/reviewed"
BLOB_NAME = "e920a7334bf6a8be7efbee247bfb4bd85ab296ed5a9629bed0d35e9da37f3d19"


@pytest.fixture
def installed(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    snapshot = cache / "models--example--reviewed" / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    blob = cache / "blobs" / BLOB_NAME[:2] / BLOB_NAME
    blob.parent.mkdir(parents=True)
    raw = json.dumps({"_class_name": "Cosmos3OmniPipeline"}).encode()
    blob.write_bytes(raw)
    marker = snapshot / "model_index.json"
    marker.symlink_to(blob)
    monkeypatch.setattr(hf, "_hf_cache_locations", lambda: [("configured", str(cache))])
    monkeypatch.setattr(hf, "cached_file_path", lambda repo, filename, revision: str(snapshot / filename))
    return cache, snapshot, marker, blob


def test_exact_snapshot_and_index_accept_shared_blob_without_content_hash_assumption(installed):
    _, snapshot, marker, blob = installed
    assert hashlib.sha256(blob.read_bytes()).hexdigest() != blob.name
    assert hf.exact_cached_snapshot_path(REPOSITORY, REVISION) == snapshot
    assert _read_reviewed_pipeline_index(marker, repository=REPOSITORY, revision=REVISION) == {
        "_class_name": "Cosmos3OmniPipeline"
    }


@pytest.mark.parametrize("target", [
    "models--foreign--repo/blobs/other", "other-managed-data", "blobs/e8/" + BLOB_NAME,
    "blobs/e9/not-a-blob", "blobs/aa/" + "A" * 64, "blobs/e9/" + "f" * 64 + "/extra",
])
def test_shared_blob_does_not_authorize_other_managed_paths(installed, target):
    cache, _, marker, _ = installed
    other = cache / target
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text("{}")
    marker.unlink()
    marker.symlink_to(other)
    with pytest.raises((ValueError, OSError), match="cache|blob|snapshot"):
        hf.exact_cached_snapshot_path(REPOSITORY, REVISION)
    with pytest.raises(OSError, match="cache|blob|snapshot"):
        _read_reviewed_pipeline_index(marker, repository=REPOSITORY, revision=REVISION)


@pytest.mark.skipif(os.name != "nt", reason="Real Windows path resolution is required for this NTFS alias check.")
def test_windows_case_variant_shared_blob_resolves_to_same_lowercase_file(installed):
    _, snapshot, marker, blob = installed
    case_alias = blob.with_name(blob.name.upper())
    assert case_alias.exists() and case_alias.samefile(blob)
    marker.unlink()
    marker.symlink_to(case_alias)
    assert marker.resolve().name == BLOB_NAME
    assert hf.exact_cached_snapshot_path(REPOSITORY, REVISION) == snapshot
    assert _read_reviewed_pipeline_index(marker, repository=REPOSITORY, revision=REVISION) == {
        "_class_name": "Cosmos3OmniPipeline"
    }


@pytest.mark.parametrize("directory", ["blobs", "blobs/e9"])
def test_shared_blob_directories_cannot_be_linked(installed, tmp_path, directory):
    cache, _, marker, blob = installed
    linked = cache / directory
    real = tmp_path / "real-blobs"
    linked.rename(real)
    linked.symlink_to(real, target_is_directory=True)
    assert marker.resolve().is_file()
    with pytest.raises((ValueError, OSError), match="cache|blob|snapshot"):
        hf.exact_cached_snapshot_path(REPOSITORY, REVISION)
    with pytest.raises(OSError, match="cache|blob|snapshot"):
        _read_reviewed_pipeline_index(marker, repository=REPOSITORY, revision=REVISION)


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'[]', b'\xff'])
def test_modern_index_keeps_strict_json_boundary(installed, raw):
    _, _, marker, blob = installed
    blob.write_bytes(raw)
    with pytest.raises(OSError, match="JSON|object"):
        _read_reviewed_pipeline_index(marker, repository=REPOSITORY, revision=REVISION)


def test_modern_index_keeps_exact_revision_and_size_limit(installed, monkeypatch):
    _, _, marker, blob = installed
    with pytest.raises(OSError, match="exact Hub snapshot"):
        _read_reviewed_pipeline_index(marker, repository=REPOSITORY, revision="b" * 40)
    monkeypatch.setattr("modules.ModularDiffusers.loaders.MAX_REVIEWED_PIPELINE_INDEX_BYTES", 10)
    with pytest.raises(OSError, match="limit"):
        _read_reviewed_pipeline_index(marker, repository=REPOSITORY, revision=REVISION)


def test_unused_linked_repository_blobs_does_not_bypass_old_directory_guard(installed):
    cache, snapshot, marker, _ = installed
    (snapshot.parent.parent / "blobs").symlink_to(cache / "blobs", target_is_directory=True)
    with pytest.raises(ValueError, match="cache"):
        hf.exact_cached_snapshot_path(REPOSITORY, REVISION)
    with pytest.raises(OSError, match="cache"):
        _read_reviewed_pipeline_index(marker, repository=REPOSITORY, revision=REVISION)


def test_foreign_repository_alias_cannot_borrow_shared_blob_containment(installed):
    cache, _, marker, blob = installed
    foreign_alias = cache / "models--foreign--repo/snapshots" / REVISION / "model_index.json"
    foreign_alias.parent.mkdir(parents=True)
    foreign_alias.symlink_to(blob)
    marker.unlink()
    marker.symlink_to(foreign_alias)
    assert marker.resolve() == blob
    with pytest.raises(ValueError, match="cache"):
        hf.exact_cached_snapshot_path(REPOSITORY, REVISION)
    with pytest.raises(OSError, match="cache"):
        _read_reviewed_pipeline_index(marker, repository=REPOSITORY, revision=REVISION)


@pytest.mark.parametrize("extended,ordinary", [
    (r"\\?\C:\hub\blobs\e9\blob", r"C:\hub\blobs\e9\blob"),
    (r"\\?\c:\hub\blobs\e9\blob", r"C:\hub\blobs\e9\blob"),
    (rf"\\?\c:\HUB\BLOBS\E9\{BLOB_NAME.upper()}", rf"C:\hub\blobs\e9\{BLOB_NAME}"),
    (r"\\?\UNC\server\share\hub\blobs\e9\blob", r"\\server\share\hub\blobs\e9\blob"),
])
def test_direct_windows_link_target_preserves_extended_path_spelling(extended, ordinary):
    from pathlib import PureWindowsPath
    from modiff.hf_cache_layout import _lexical_cache_path
    # os.readlink exposes the substitute path; Path.resolve removes this prefix
    # for a nonextended alias. Compare lexical identities, never follow links.
    direct, resolved = PureWindowsPath(extended), PureWindowsPath(ordinary)
    assert direct != resolved  # The previous equality guard rejected this valid target.
    assert _lexical_cache_path(direct) == _lexical_cache_path(resolved)


@pytest.mark.parametrize("other", [
    r"\\?\C:\hub\models--foreign--repo\snapshots\a\model_index.json",
    r"\\?\C:\hub\blobs\e9\other",
    r"\\?\D:\hub\blobs\e9\blob",
    r"\\?\GLOBALROOT\Device\hub\blobs\e9\blob",
    r"\\.\C:\hub\blobs\e9\blob",
    r"\\?\C:relative",
])
def test_windows_prefix_normalization_does_not_authorize_other_targets(other):
    from pathlib import PureWindowsPath
    from modiff.hf_cache_layout import _lexical_cache_path
    assert _lexical_cache_path(PureWindowsPath(other)) != PureWindowsPath(r"C:\hub\blobs\e9\blob")


def test_windows_prefix_normalization_leaves_posix_path_unchanged():
    from modiff.hf_cache_layout import _lexical_cache_path
    path = PurePosixPath("/hub") / r"\\?\C:\literal"
    assert _lexical_cache_path(path) is path
