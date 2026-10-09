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


def repository_blob_bridge(installed):
    cache, snapshot, marker, blob = installed
    bridge = snapshot.parent.parent / "blobs" / ("b" * 40)
    bridge.parent.mkdir()
    bridge.symlink_to(os.path.relpath(blob, bridge.parent))
    marker.unlink()
    marker.symlink_to(os.path.relpath(bridge, marker.parent))
    return cache, snapshot, marker, blob, bridge


def test_migrated_repository_blob_bridge_keeps_exact_snapshot_and_index(installed):
    _, snapshot, marker, _, _ = repository_blob_bridge(installed)
    assert hf.exact_cached_snapshot_path(REPOSITORY, REVISION) == snapshot
    assert _read_reviewed_pipeline_index(marker, repository=REPOSITORY, revision=REVISION) == {
        "_class_name": "Cosmos3OmniPipeline"
    }


def test_migrated_blob_bridge_accepts_nested_snapshot_files(installed):
    from modiff.hf_cache_layout import resolve_snapshot_cache_file
    cache, snapshot, _, blob, bridge = repository_blob_bridge(installed)
    alias = snapshot / "text_tokenizer" / "tokenizer.json"
    alias.parent.mkdir()
    alias.symlink_to(os.path.relpath(bridge, alias.parent))
    assert resolve_snapshot_cache_file(
        alias, snapshot=snapshot, cache_root=cache, repository=REPOSITORY,
    ) == blob


@pytest.mark.parametrize("fault", [
    "foreign_repository", "extra_repository_hop", "foreign_alias_hop", "shared_file_hop",
    "linked_repository_blobs", "linked_shared_shard", "wrong_shared_shard", "nested_repository_blob",
    "own_regular_file", "invalid_repository_blob_name",
])
def test_migrated_blob_bridge_does_not_authorize_extra_links_or_directories(installed, tmp_path, fault):
    cache, snapshot, marker, blob, bridge = repository_blob_bridge(installed)
    if fault == "foreign_repository":
        target = cache / "models--foreign--repo/blobs" / bridge.name
        target.parent.mkdir(parents=True)
        target.symlink_to(blob)
        marker.unlink()
        marker.symlink_to(target)
    elif fault in {"extra_repository_hop", "foreign_alias_hop", "shared_file_hop"}:
        target = {
            "extra_repository_hop": bridge.parent / ("c" * 40),
            "foreign_alias_hop": cache / "models--foreign--repo/blobs" / ("c" * 40),
            "shared_file_hop": cache / "blobs/e9" / ("e9" + "c" * 62),
        }[fault]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(blob)
        bridge.unlink()
        bridge.symlink_to(target)
    elif fault in {"linked_repository_blobs", "linked_shared_shard"}:
        directory = bridge.parent if fault == "linked_repository_blobs" else blob.parent
        real = tmp_path / "real-blob-directory"
        directory.rename(real)
        directory.symlink_to(real, target_is_directory=True)
        if fault == "linked_repository_blobs":
            moved_bridge = real / bridge.name
            moved_bridge.unlink()
            moved_bridge.symlink_to(blob)
    elif fault == "wrong_shared_shard":
        target = cache / "blobs/aa" / BLOB_NAME
        target.parent.mkdir()
        target.write_bytes(blob.read_bytes())
        bridge.unlink()
        bridge.symlink_to(target)
    elif fault == "nested_repository_blob":
        target = bridge.parent / "nested" / bridge.name
        target.parent.mkdir()
        target.symlink_to(blob)
        marker.unlink()
        marker.symlink_to(target)
    elif fault == "own_regular_file":
        target = bridge.parent / ("c" * 40)
        target.write_bytes(blob.read_bytes())
        bridge.unlink()
        bridge.symlink_to(target)
    else:
        target = bridge.with_name("not-an-etag")
        target.symlink_to(blob)
        marker.unlink()
        marker.symlink_to(target)
    with pytest.raises((ValueError, OSError), match="cache|blob|snapshot"):
        hf.exact_cached_snapshot_path(REPOSITORY, REVISION)
    with pytest.raises(OSError, match="cache|blob|snapshot"):
        _read_reviewed_pipeline_index(marker, repository=REPOSITORY, revision=REVISION)


def test_invalid_cache_layout_has_precise_wrapped_loader_diagnostic(installed):
    from modiff.server import WebServer
    cache, _, marker, blob, _ = repository_blob_bridge(installed)
    foreign = cache / "models--foreign--repo/blobs" / ("d" * 40)
    foreign.parent.mkdir(parents=True)
    foreign.symlink_to(blob)
    marker.unlink()
    marker.symlink_to(foreign)
    with pytest.raises(OSError) as caught:
        _read_reviewed_pipeline_index(marker, repository=REPOSITORY, revision=REVISION)
    wrapped = RuntimeError("Error executing modules.ModularDiffusers.ModelsLoader")
    wrapped.__cause__ = caught.value
    result = WebServer._classify_exception(object.__new__(WebServer), wrapped)
    assert result["category"] == "model_integrity"
    assert result["error_code"] == "invalid_model_cache_layout"
    assert "blob" in result["message"]
    assert "Model Manager" in result["recovery_hint"]
    assert "remains runnable" not in result["recovery_hint"]
    ordinary = WebServer._classify_exception(object.__new__(WebServer), ValueError("Width must be positive"))
    assert ordinary["error_code"] == "invalid_node_input"


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


@pytest.mark.parametrize("root,target", [
    (r"C:\hub\models--example--reviewed\blobs", rf"\\?\C:\hub\models--example--reviewed\blobs\{'b' * 40}"),
    (r"C:\hub\models--example--reviewed\blobs", rf"\\?\c:\HUB\MODELS--EXAMPLE--REVIEWED\BLOBS\{'b' * 40}"),
    (r"\\server\share\hub\models--example--reviewed\blobs", rf"\\?\UNC\server\share\hub\models--example--reviewed\blobs\{'b' * 64}"),
])
def test_windows_migrated_blob_bridge_owns_extended_and_case_variant_paths(root, target):
    from pathlib import PureWindowsPath
    from modiff.hf_cache_layout import _flat_repository_blob_path
    # readlink's substitute-path prefix differs from the concrete cache root.
    # This is the old production relative_to operation, which rejected the link.
    with pytest.raises(ValueError):
        PureWindowsPath(target).relative_to(PureWindowsPath(root))
    assert _flat_repository_blob_path(PureWindowsPath(target), PureWindowsPath(root)).name in {
        "b" * 40, "b" * 64,
    }


@pytest.mark.parametrize("target", [
    rf"\\?\C:\hub\models--foreign--repo\blobs\{'b' * 40}",
    rf"\\?\D:\hub\models--example--reviewed\blobs\{'b' * 40}",
    rf"\\?\C:\hub\models--example--reviewed\blobs\nested\{'b' * 40}",
    r"\\?\C:\hub\models--example--reviewed\blobs\not-an-etag",
    rf"\\?\UNC\other\share\hub\models--example--reviewed\blobs\{'b' * 40}",
])
def test_windows_migrated_blob_bridge_rejects_foreign_drives_repositories_and_nested_paths(target):
    from pathlib import PureWindowsPath
    from modiff.hf_cache_layout import HuggingFaceCacheLayoutError, _flat_repository_blob_path
    with pytest.raises(HuggingFaceCacheLayoutError):
        _flat_repository_blob_path(PureWindowsPath(target), PureWindowsPath(r"C:\hub\models--example--reviewed\blobs"))
