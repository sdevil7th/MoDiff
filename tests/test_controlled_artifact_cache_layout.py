"""Artifact entry points must share execution's finite Hub cache layout contract."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from modiff.auxiliary_lora import build_lora_descriptor, resolve_lora_descriptor
from modiff.controlled_artifacts import controlled_artifact_receipts_from_graph, resolve_upscaler_artifact
from utils import huggingface as hf


REPOSITORY = "example/style"
REVISION = "a" * 40
WEIGHT_NAME = "weights/style.safetensors"
BLOB_ID = "e9" + "b" * 62


@pytest.fixture
def installed(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    repository = cache / "models--example--style"
    snapshot = repository / "snapshots" / REVISION
    alias = snapshot / WEIGHT_NAME
    blob = cache / "blobs" / BLOB_ID[:2] / BLOB_ID
    alias.parent.mkdir(parents=True)
    blob.parent.mkdir(parents=True)
    save_file({"lora.weight": np.asarray([1.0], dtype=np.float32)}, str(blob))
    digest = hashlib.sha256(blob.read_bytes()).hexdigest()
    assert digest != BLOB_ID  # Hub storage identity is not the artifact's digest.
    monkeypatch.setitem(hf.CONFIG.hf, "cache_dir", str(cache))
    monkeypatch.setattr(hf, "cached_file_path", lambda *args, **kwargs: str(alias))
    return {
        "cache": cache,
        "repository": repository,
        "snapshot": snapshot,
        "alias": alias,
        "blob": blob,
        "digest": digest,
        "size": blob.stat().st_size,
    }


def _link(alias, target):
    alias.symlink_to(os.path.relpath(target, alias.parent))


def _resolve(installed, consumer, *, digest=None, size=None):
    digest = installed["digest"] if digest is None else digest
    if consumer == "lora":
        descriptor = build_lora_descriptor(
            selection={"source": "hub", "value": REPOSITORY},
            weight_name=WEIGHT_NAME,
            revision=REVISION,
            expected_sha256=digest,
            adapter_name="style",
            scale=0.5,
        )
        # Exercise the consumer's independent revalidation, not only authoring.
        resolved = resolve_lora_descriptor(descriptor)
        assert resolved.load_directory == installed["alias"].parent
        assert resolved.weight_name == "style.safetensors"
        return descriptor
    resolved = resolve_upscaler_artifact(
        {
            "source": "hub",
            "value": f"{REPOSITORY}/{WEIGHT_NAME}",
            "revision": REVISION,
            "sha256": digest,
            "byteSize": installed["size"] if size is None else size,
        }
    )
    assert resolved.path == installed["alias"].resolve(strict=True)
    return resolved.receipt


@pytest.mark.parametrize("consumer", ["lora", "upscaler"])
@pytest.mark.parametrize("layout", ["ordinary", "repository_blob", "shared_blob", "repository_bridge"])
def test_artifact_entry_points_accept_reviewed_layouts_and_hash_actual_bytes(installed, consumer, layout):
    alias, blob = installed["alias"], installed["blob"]
    if layout == "ordinary":
        alias.write_bytes(blob.read_bytes())
    elif layout == "repository_blob":
        own_blob = installed["repository"] / "blobs" / ("c" * 64)
        own_blob.parent.mkdir()
        own_blob.write_bytes(blob.read_bytes())
        _link(alias, own_blob)
    elif layout == "shared_blob":
        _link(alias, blob)
    else:
        bridge = installed["repository"] / "blobs" / ("c" * 40)
        bridge.parent.mkdir()
        _link(bridge, blob)
        _link(alias, bridge)

    receipt = _resolve(installed, consumer)

    assert receipt["artifact"]["sha256"] == installed["digest"]
    assert receipt["artifact"]["revision"] == REVISION
    assert str(installed["cache"]) not in json.dumps(receipt)


@pytest.mark.parametrize(
    "module,action",
    [
        ("modules.ModularDiffusers", "Lora"),
        ("modules.DiffusersImage", "LoadAdapter"),
        ("modules.Spandrel", "Upscaler"),
    ],
)
def test_executable_graph_receipt_resolves_an_owned_shared_blob_bridge(installed, module, action):
    bridge = installed["repository"] / "blobs" / ("c" * 40)
    bridge.parent.mkdir()
    _link(bridge, installed["blob"])
    _link(installed["alias"], bridge)
    if action == "Upscaler":
        params = {
            "model_id": {
                "source": "hub",
                "value": f"{REPOSITORY}/{WEIGHT_NAME}",
                "revision": REVISION,
                "sha256": installed["digest"],
                "byteSize": installed["size"],
            }
        }
    else:
        params = {
            "model" if action == "Lora" else "adapter_path": {"source": "hub", "value": REPOSITORY},
            "weight_name": WEIGHT_NAME,
            "revision": REVISION,
            "expected_sha256": installed["digest"],
            "adapter_name": "style",
            "scale": 0.5,
        }
    node = {"module": module, "action": action, "params": {key: {"value": value} for key, value in params.items()}}
    graph = {"nodes": {"artifact": node, "disconnected": node}, "paths": [["artifact"]]}

    receipts = controlled_artifact_receipts_from_graph(graph)

    assert len(receipts) == 1
    assert receipts[0]["module"] == module
    assert receipts[0]["action"] == action
    assert receipts[0]["artifact"]["sha256"] == installed["digest"]
    assert receipts[0]["artifact"]["revision"] == REVISION
    assert str(installed["cache"]) not in json.dumps(receipts)


@pytest.mark.parametrize("consumer", ["lora", "upscaler"])
@pytest.mark.parametrize(
    "fault",
    [
        "outside_cache",
        "foreign_repository_blob",
        "foreign_snapshot_alias",
        "same_repository_snapshot_alias",
        "extra_repository_hop",
        "shared_file_hop",
        "nested_repository_bridge",
        "invalid_repository_bridge",
        "repository_regular_file_hop",
        "linked_repository_blobs",
        "linked_shared_store",
        "linked_shared_shard",
        "wrong_shared_shard",
        "invalid_shared_name",
        "nested_shared_blob",
        "linked_snapshot_directory",
        "linked_snapshot",
        "linked_alias_parent",
    ],
)
def test_artifact_entry_points_reject_foreign_and_malformed_cache_links(installed, tmp_path, consumer, fault):
    alias, blob = installed["alias"], installed["blob"]
    repository = installed["repository"]
    target = blob
    if fault == "outside_cache":
        target = tmp_path / "outside.safetensors"
        target.write_bytes(blob.read_bytes())
    elif fault in {"foreign_repository_blob", "foreign_snapshot_alias", "same_repository_snapshot_alias"}:
        target = {
            "foreign_repository_blob": installed["cache"] / "models--foreign--repo" / "blobs" / ("c" * 40),
            "foreign_snapshot_alias": installed["cache"]
            / "models--foreign--repo"
            / "snapshots"
            / REVISION
            / "other.safetensors",
            "same_repository_snapshot_alias": repository / "snapshots" / ("d" * 40) / "other.safetensors",
        }[fault]
        target.parent.mkdir(parents=True)
        _link(target, blob)
    elif fault in {"extra_repository_hop", "shared_file_hop", "repository_regular_file_hop"}:
        target = repository / "blobs" / ("c" * 40)
        target.parent.mkdir()
        second = blob.parent / ("e9" + "d" * 62) if fault == "shared_file_hop" else target.parent / ("d" * 40)
        if fault == "repository_regular_file_hop":
            second.write_bytes(blob.read_bytes())
        else:
            _link(second, blob)
        _link(target, second)
    elif fault in {"nested_repository_bridge", "invalid_repository_bridge"}:
        target = (
            repository / "blobs" / (f"nested/{'c' * 40}" if fault == "nested_repository_bridge" else "not-an-etag")
        )
        target.parent.mkdir(parents=True)
        _link(target, blob)
    elif fault in {"linked_repository_blobs", "linked_shared_store", "linked_shared_shard"}:
        directory = {
            "linked_repository_blobs": repository / "blobs",
            "linked_shared_store": blob.parent.parent,
            "linked_shared_shard": blob.parent,
        }[fault]
        if fault == "linked_repository_blobs":
            directory.mkdir()
        real_directory = tmp_path / "redirected-directory"
        directory.rename(real_directory)
        directory.symlink_to(real_directory, target_is_directory=True)
    elif fault in {"wrong_shared_shard", "invalid_shared_name", "nested_shared_blob"}:
        target = {
            "wrong_shared_shard": installed["cache"] / "blobs" / "aa" / BLOB_ID,
            "invalid_shared_name": blob.with_name("not-a-storage-identity"),
            "nested_shared_blob": blob.parent / "nested" / BLOB_ID,
        }[fault]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob.read_bytes())
    else:
        directory = {
            "linked_snapshot_directory": installed["snapshot"].parent,
            "linked_snapshot": installed["snapshot"],
            "linked_alias_parent": alias.parent,
        }[fault]
        real_directory = tmp_path / "redirected-directory"
        directory.rename(real_directory)
        directory.symlink_to(real_directory, target_is_directory=True)
    _link(alias, target)

    with pytest.raises(ValueError, match="managed|snapshot|cache"):
        _resolve(installed, consumer)


@pytest.mark.parametrize("consumer", ["lora", "upscaler"])
@pytest.mark.parametrize("lookup", ["foreign_repository", "different_revision", "different_name", "relative_alias"])
def test_artifact_entry_points_preserve_exact_lexical_cache_lookup(installed, monkeypatch, consumer, lookup):
    _link(installed["alias"], installed["blob"])
    returned = {
        "foreign_repository": installed["cache"] / "models--foreign--repo" / "snapshots" / REVISION / WEIGHT_NAME,
        "different_revision": installed["repository"] / "snapshots" / ("d" * 40) / WEIGHT_NAME,
        "different_name": installed["alias"].with_name("other.safetensors"),
        "relative_alias": Path("models--example--style") / "snapshots" / REVISION / WEIGHT_NAME,
    }[lookup]
    monkeypatch.setattr(hf, "cached_file_path", lambda *args, **kwargs: str(returned))
    with pytest.raises(ValueError, match="absolute|snapshot"):
        _resolve(installed, consumer)


@pytest.mark.parametrize("consumer", ["lora", "upscaler"])
def test_shared_blob_storage_identity_cannot_replace_the_content_digest(installed, consumer):
    _link(installed["alias"], installed["blob"])
    with pytest.raises(ValueError, match="SHA-256"):
        _resolve(installed, consumer, digest=BLOB_ID)


@pytest.mark.parametrize("consumer", ["lora", "upscaler"])
def test_shared_blob_content_is_revalidated_before_consumption(installed, consumer):
    _link(installed["alias"], installed["blob"])
    receipt = _resolve(installed, consumer)
    save_file({"lora.weight": np.asarray([2.0], dtype=np.float32)}, str(installed["blob"]))
    with pytest.raises(ValueError, match="SHA-256"):
        if consumer == "lora":
            resolve_lora_descriptor(receipt)
        else:
            _resolve(installed, consumer)


def test_shared_lora_blob_still_requires_a_valid_safetensors_header(installed):
    _link(installed["alias"], installed["blob"])
    installed["blob"].write_bytes(b"not a Safetensors header")
    digest = hashlib.sha256(installed["blob"].read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="valid Safetensors"):
        _resolve(installed, "lora", digest=digest)


def test_shared_upscaler_blob_still_requires_the_declared_size(installed):
    _link(installed["alias"], installed["blob"])
    with pytest.raises(ValueError, match="declared size"):
        _resolve(installed, "upscaler", size=installed["size"] + 1)
