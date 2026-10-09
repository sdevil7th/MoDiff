import json
import struct

import pytest

from modiff.artifact_weight_memory import artifact_parameter_lower_bound, impossible_parameter_residency


REVISION = "a" * 40


def write_weights(path, tensors):
    offsets = 0
    header = {}
    for name, (dtype, shape, size) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offsets, offsets + size]}
        offsets += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offsets))


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "models--test--model" / "snapshots" / REVISION
    root.mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({"transformer": ["diffusers", "Transformer"], "text_encoder": ["transformers", "Encoder"]}))
    for name in ("transformer", "text_encoder"):
        (root / name).mkdir()
    write_weights(root / "transformer" / "weights.safetensors", {"matrix": ("F32", [3, 4], 48)})
    write_weights(root / "text_encoder" / "weights.safetensors", {"matrix": ("BF16", [2, 3], 12)})
    status = {"complete": True, "snapshots": [{"complete": True, "path": str(root)}]}
    return root, status


def test_dense_storage_counts_requested_dtype_and_components_without_importing_models(snapshot):
    _, status = snapshot
    result = artifact_parameter_lower_bound(status, REVISION, "bfloat16")
    assert result["components"] == {"transformer": 24, "text_encoder": 12}
    assert result["totalBytes"] == 36
    assert result["largestComponentBytes"] == 24
    assert result["proofScope"] == "parameter_storage_lower_bound_only"
    assert artifact_parameter_lower_bound(status, REVISION, "float32")["totalBytes"] == 72


def test_sharded_indexes_count_each_selected_tensor_once(snapshot):
    root, status = snapshot
    folder = root / "transformer"
    write_weights(folder / "weights.safetensors", {"one": ("F32", [2], 8), "two": ("F32", [2], 8), "unused": ("F32", [20], 80)})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"one": "weights.safetensors", "two": "weights.safetensors"}}))
    result = artifact_parameter_lower_bound(status, REVISION, "bfloat16")
    assert result["components"]["transformer"] == 8


@pytest.mark.parametrize("revision,dtype,complete", [("b" * 40, "bfloat16", True), ("main", "bfloat16", True), (REVISION, "int4", True), (REVISION, "float32", False)])
def test_no_bound_from_wrong_revision_unsupported_conversion_or_incomplete_snapshot(snapshot, revision, dtype, complete):
    _, status = snapshot
    status["complete"] = complete
    assert artifact_parameter_lower_bound(status, revision, dtype) is None


@pytest.mark.parametrize("filename", ["../weights.safetensors", "..\\weights.safetensors", "/weights.safetensors", "C:weights.safetensors"])
def test_weight_index_cannot_escape_the_cache(snapshot, filename):
    root, status = snapshot
    (root / "transformer" / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"matrix": filename}}))
    assert artifact_parameter_lower_bound(status, REVISION, "bfloat16") is None


def test_oversized_or_inconsistent_tensor_headers_do_not_supply_a_bound(snapshot):
    root, status = snapshot
    path = root / "transformer" / "weights.safetensors"
    path.write_bytes(struct.pack("<Q", 16 * 1024 * 1024 + 1))
    assert artifact_parameter_lower_bound(status, REVISION, "bfloat16") is None
    write_weights(path, {"matrix": ("F32", [100], 48)})
    assert artifact_parameter_lower_bound(status, REVISION, "bfloat16") is None


@pytest.mark.parametrize("mode,blocked", [("none", True), ("model_cpu", True), ("group_cpu", False), ("group_disk", False)])
def test_whole_component_gpu_residency_is_rejected_even_when_the_old_machine_tier_fits(mode, blocked):
    gib = 1024**3
    hardware = {"accelerator": {"kind": "cuda", "totalBytes": 16*gib}}
    bound = {"totalBytes": 54*gib, "largestComponentBytes": 40*gib}
    missing = impossible_parameter_residency(hardware, bound, mode)
    assert bool(missing) == blocked
    if blocked:
        assert "parameter storage" in missing[0]


def test_no_lower_bound_is_not_a_new_fit_or_qualification_claim():
    assert impossible_parameter_residency({"accelerator": {"kind": "cuda", "totalBytes": 1}}, None, "none") == []


def test_final_candidate_residency_cannot_be_promoted_past_a_known_parameter_bound(monkeypatch, tmp_path):
    from modiff import auto_resource as planner

    gib = 1024**3
    candidate = {
        "id": "native", "modelType": "ZImageModularPipeline", "mode": "modular_text_to_image",
        "artifactFormat": "native", "artifact": planner.Z_IMAGE_REPO,
        "modelRepo": planner.Z_IMAGE_REPO, "resolvedArtifact": planner.Z_IMAGE_REPO,
        "dtype": "bfloat16", "quantizationMode": "none", "loadedQuantization": "none",
        "offloadMode": "group_disk", "installed": True, "artifactValidation": {"complete": True},
        "artifactResolution": {"resolved": {"repo": planner.Z_IMAGE_REPO, "revision": REVISION}},
        "proof": {"status": "live_proven"}, "canAutoRun": True, "requirementsMissing": [],
        "qualityTier": "native", "qualityScore": 100, "rank": 1,
    }
    monkeypatch.setattr(planner, "_declared_profile_candidates", lambda *args: [candidate.copy()])
    monkeypatch.setattr(planner, "qualified_auto_overrides", lambda **kwargs: {"offloadMode": "model_cpu"})
    monkeypatch.setattr(planner, "artifact_parameter_lower_bound", lambda *args: {"totalBytes": 54*gib, "largestComponentBytes": 40*gib})
    hardware = {"accelerator": {"kind": "cuda", "totalBytes": 16*gib, "freeBytes": 16*gib},
                "systemMemory": {"totalBytes": 32*gib, "availableBytes": 32*gib}, "offloadDisk": {"freeBytes": 100*gib}}
    result = planner.build_auto_resource_plan(
        {"form": {"modelType": "ZImageModularPipeline", "mode": "modular_text_to_image"}, "hardwareOverride": hardware},
        runtime_fingerprint={}, local_models=[], data_dir=tmp_path, history={"entries": {}},
    )
    assert result["status"] != "ready"
    assert result["selectedCandidate"] is None
    assert result["candidates"][0]["canAutoRun"] is False
    assert result["candidates"][0]["proof"]["status"] == "skipped"
    assert "parameter storage" in result["candidates"][0]["skipReason"]
