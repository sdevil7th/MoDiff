"""Model-free cache-root checks for controlled LoRA and upscaler boundaries."""

import hashlib
from unittest.mock import Mock

import numpy as np
import pytest
from safetensors.numpy import save_file

from modiff.auxiliary_lora import build_lora_descriptor, controlled_lora_receipts_from_graph, resolve_lora_descriptor
from modiff.controlled_artifacts import (
    controlled_artifact_receipts_from_graph,
    pin_upscaler_model_selection,
    resolve_upscaler_artifact,
)
from utils import huggingface as hf


REPOSITORY = "example/style"
REVISION = "a" * 40
WEIGHT_NAME = "weights/style.safetensors"


@pytest.fixture
def secondary_cache(tmp_path, monkeypatch):
    configured = tmp_path / "private" / "hub"
    secondary = tmp_path / "discovery" / "hub"
    configured.mkdir(parents=True)
    repository = secondary / "models--example--style"
    alias = repository / "snapshots" / REVISION / WEIGHT_NAME
    blob = repository / "blobs" / ("b" * 64)
    alias.parent.mkdir(parents=True)
    blob.parent.mkdir()
    save_file({"lora.weight": np.asarray([1.0], dtype=np.float32)}, str(blob))
    alias.symlink_to(blob)
    digest = hashlib.sha256(blob.read_bytes()).hexdigest()
    locations = [("Configured", str(configured)), ("TRANSFORMERS_CACHE", str(secondary))]
    monkeypatch.setitem(hf.CONFIG.hf, "cache_dir", str(configured))
    monkeypatch.setattr(hf, "_hf_cache_locations", lambda: locations)
    params = {
        "model": {"source": "hub", "value": REPOSITORY},
        "weight_name": WEIGHT_NAME,
        "revision": REVISION,
        "expected_sha256": digest,
        "adapter_name": "style",
        "scale": 0.5,
    }
    graph = {
        "nodes": {"adapter": {
            "module": "modules.ModularDiffusers", "action": "Lora",
            "params": {key: {"value": value} for key, value in params.items()},
        }},
        "paths": [["adapter"]],
    }
    return {"configured": configured, "secondary": secondary, "repository": repository,
            "alias": alias, "blob": blob, "digest": digest, "graph": graph}


def test_controlled_lora_accepts_an_approved_secondary_cache_without_a_mutable_ref(secondary_cache):
    installed = secondary_cache

    receipts = controlled_lora_receipts_from_graph(installed["graph"], maximum_weight_bytes=1024)

    assert len(receipts) == 1
    assert receipts[0]["artifact"]["sha256"] == installed["digest"]
    assert receipts[0]["artifact"]["revision"] == REVISION
    assert not (installed["repository"] / "refs").exists()
    assert not list(installed["configured"].iterdir())
    descriptor = build_lora_descriptor(
        selection={"source": "hub", "value": REPOSITORY}, weight_name=WEIGHT_NAME,
        revision=REVISION, expected_sha256=installed["digest"], adapter_name="style", scale=0.5,
        maximum_weight_bytes=1024,
    )
    assert receipts[0]["descriptorSha256"] == descriptor["descriptor_sha256"]
    resolved = resolve_lora_descriptor(descriptor)
    assert resolved.load_directory == installed["alias"].parent
    assert resolved.weight_name == "style.safetensors"


def _upscaler_selection(installed):
    return {"source": "hub", "value": f"{REPOSITORY}/{WEIGHT_NAME}", "revision": REVISION,
            "sha256": installed["digest"], "byteSize": installed["blob"].stat().st_size}


def _upscaler_graph(selection):
    return {"nodes": {"upscaler": {"module": "modules.Spandrel", "action": "Upscaler",
                                  "params": {"model_id": {"value": selection}}}},
            "paths": [["upscaler"]]}


def test_controlled_upscaler_accepts_an_approved_secondary_cache_without_a_mutable_ref(secondary_cache):
    installed = secondary_cache
    expected = _upscaler_selection(installed)

    pinned = pin_upscaler_model_selection({key: value for key, value in expected.items()
                                          if key not in {"sha256", "byteSize"}})
    receipts = controlled_artifact_receipts_from_graph(_upscaler_graph(pinned))
    resolved = resolve_upscaler_artifact(pinned)

    assert pinned == expected
    assert len(receipts) == 1
    assert receipts[0]["artifact"]["sha256"] == installed["digest"]
    assert receipts[0]["artifact"]["revision"] == REVISION
    assert resolved.path == installed["blob"]
    assert not (installed["repository"] / "refs").exists()
    assert not list(installed["configured"].iterdir())


@pytest.mark.parametrize("consumer", ["lora", "upscaler"])
@pytest.mark.parametrize("fault", [
    "unapproved_root", "unapproved_link_to_approved_alias", "wrong_repository", "wrong_revision",
    "nested_cache_path", "outside_blob", "other_approved_cache_blob", "foreign_repository_blob",
])
def test_controlled_artifact_rejects_alias_and_blob_escapes_before_hashing(
    secondary_cache, tmp_path, monkeypatch, consumer, fault,
):
    installed = secondary_cache
    alias = installed["alias"]
    if fault in {"unapproved_root", "unapproved_link_to_approved_alias"}:
        alias = tmp_path / "unapproved" / "hub" / "models--example--style" / "snapshots" / REVISION / WEIGHT_NAME
    elif fault == "wrong_repository":
        alias = installed["secondary"] / "models--other--style" / "snapshots" / REVISION / WEIGHT_NAME
    elif fault == "wrong_revision":
        alias = installed["repository"] / "snapshots" / ("c" * 40) / WEIGHT_NAME
    elif fault == "nested_cache_path":
        alias = installed["secondary"] / "nested" / "models--example--style" / "snapshots" / REVISION / WEIGHT_NAME

    if alias != installed["alias"]:
        alias.parent.mkdir(parents=True)
        alias.symlink_to(installed["alias"] if fault == "unapproved_link_to_approved_alias" else installed["blob"])
        message = "exact repository snapshot path" if consumer == "lora" else "exact repository snapshot alias"
    else:
        target = {
            "outside_blob": tmp_path / "outside.safetensors",
            "other_approved_cache_blob": installed["configured"] / "models--example--style" / "blobs" / ("c" * 64),
            "foreign_repository_blob": installed["secondary"] / "models--other--style" / "blobs" / ("c" * 64),
        }[fault]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(installed["blob"].read_bytes())
        alias.unlink()
        alias.symlink_to(target)
        message = "outside its managed repository cache" if consumer == "lora" else "outside its exact managed snapshot"
    monkeypatch.setattr(hf, "cached_file_path", lambda *args, **kwargs: str(alias))
    hash_weight = Mock(side_effect=AssertionError("Rejected aliases must not reach weight hashing."))
    monkeypatch.setattr("modiff.auxiliary_lora._sha256_file", hash_weight)
    monkeypatch.setattr("modiff.controlled_artifacts._sha256_file", hash_weight)

    if consumer == "lora":
        with pytest.raises(ValueError, match=message):
            controlled_lora_receipts_from_graph(installed["graph"], maximum_weight_bytes=1024)
    else:
        selection = _upscaler_selection(installed)
        with pytest.raises(ValueError, match=message):
            controlled_artifact_receipts_from_graph(_upscaler_graph(selection))
        with pytest.raises(ValueError, match=message):
            pin_upscaler_model_selection({key: value for key, value in selection.items() if key != "byteSize"})

    hash_weight.assert_not_called()


@pytest.mark.parametrize("consumer", ["lora", "upscaler"])
def test_controlled_artifact_does_not_bypass_a_configured_pin_failure_for_a_secondary_copy(secondary_cache, consumer):
    installed = secondary_cache
    alias = installed["configured"] / "models--example--style" / "snapshots" / REVISION / WEIGHT_NAME
    alias.parent.mkdir(parents=True)
    save_file({"lora.weight": np.asarray([2.0], dtype=np.float32)}, str(alias))
    assert hashlib.sha256(alias.read_bytes()).hexdigest() != installed["digest"]

    if consumer == "lora":
        with pytest.raises(ValueError, match="pinned SHA-256 verification"):
            controlled_lora_receipts_from_graph(installed["graph"], maximum_weight_bytes=1024)
    else:
        selection = _upscaler_selection(installed)
        with pytest.raises(ValueError, match="declared SHA-256"):
            controlled_artifact_receipts_from_graph(_upscaler_graph(selection))
        with pytest.raises(ValueError, match="declared SHA-256"):
            pin_upscaler_model_selection({key: value for key, value in selection.items() if key != "byteSize"})
