import json
import struct

import pytest

from modiff.artifact_weight_memory import (
    artifact_parameter_lower_bound, impossible_parameter_residency, native_cpu_parameter_residency,
    full_residency_capacity_missing, single_file_parameter_lower_bound,
)


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
    root = tmp_path / "hub" / "models--test--model" / "snapshots" / REVISION
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


def test_single_file_dense_floor_promotes_float_storage_but_retains_nonfloat_buffers(snapshot):
    root, _status = snapshot
    write_weights(root / 'model.safetensors', {
        'weight': ('BF16', [4, 4], 32), 'counter': ('I64', [2], 16),
        'mask': ('BOOL', [3], 3),
    })
    bound = single_file_parameter_lower_bound(root, REVISION, 'float32')
    assert bound['totalBytes'] == bound['largestComponentBytes'] == 64 + 16 + 3
    assert bound['proofScope'] == 'parameter_storage_lower_bound_only'
    assert single_file_parameter_lower_bound(root, REVISION, 'bfloat16')['totalBytes'] == 32 + 16 + 3
    assert single_file_parameter_lower_bound(root, 'main', 'float32') is None
    assert single_file_parameter_lower_bound(root, REVISION, 'unknown') is None


@pytest.mark.parametrize('malformed', ['duplicate', 'overlap'])
def test_single_file_dense_floor_rejects_ambiguous_tensor_metadata(snapshot, malformed):
    root, _status = snapshot
    tensor = '{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}'
    # Equal-sized overlapping intervals and duplicate object keys must not
    # become a proven lower bound, even if payload size is plausible.
    key = 'weight' if malformed == 'duplicate' else 'other'
    raw = ('{"weight":' + tensor + ',"' + key + '":' + tensor + '}').encode()
    (root / 'model.safetensors').write_bytes(struct.pack('<Q', len(raw)) + raw + bytes(8))
    assert single_file_parameter_lower_bound(root, REVISION, 'float32') is None


@pytest.mark.parametrize('kind', ['cuda', 'xpu', 'mps'])
def test_dense_physical_capacity_uses_exact_selected_device_and_unified_pool(kind):
    bound = {'proofScope': 'parameter_storage_lower_bound_only', 'totalBytes': 100}
    hardware = {'system': {'ram_total': 1000}, 'devices': [
        {'type': kind, 'index': 0, 'memory_kind': 'dedicated', 'dedicated_memory_total': 1000},
        {'type': kind, 'index': 1, 'memory_kind': 'dedicated', 'dedicated_memory_total': 50},
    ]}
    assert full_residency_capacity_missing(hardware, bound, kind + ':0') == []
    assert 'selected-device' in full_residency_capacity_missing(hardware, bound, kind + ':1')[0]
    hardware['devices'][1].update(memory_kind='shared', shared_memory_total=50, torch_vram_total=50)
    # ROCm APUs/MPS/XPU recommendations and planning caps are not physical
    # bounds: only system RAM can establish impossibility for a shared pool.
    assert full_residency_capacity_missing(hardware, bound, kind + ':1') == []
    hardware['system']['ram_total'] = 50
    assert full_residency_capacity_missing(hardware, bound, kind + ':1')


def test_dense_physical_capacity_refuses_impossible_cpu_loading_without_claiming_fit():
    bound = {'proofScope': 'parameter_storage_lower_bound_only', 'totalBytes': 100}
    assert full_residency_capacity_missing({'system': {'ram_total': 50}}, bound, 'cpu:0')
    # Total capacity, not free working memory: passing this floor is not fit.
    assert full_residency_capacity_missing({'system': {'ram_total': 100, 'ram_available': 1}}, bound, 'cpu:0') == []
    assert full_residency_capacity_missing({'system': {}}, bound, 'cuda:0') == []


def test_sharded_indexes_count_each_selected_tensor_once(snapshot):
    root, status = snapshot
    folder = root / "transformer"
    write_weights(folder / "weights.safetensors", {"one": ("F32", [2], 8), "two": ("F32", [2], 8), "unused": ("F32", [20], 80)})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"one": "weights.safetensors", "two": "weights.safetensors"}}))
    result = artifact_parameter_lower_bound(status, REVISION, "bfloat16")
    assert result["components"]["transformer"] == 8


def shared_blob_alias(alias, cache_root, number):
    # Hub storage identities need not be hashes of the file bytes.
    name = f"{number:064x}"
    target = cache_root / "blobs" / name[:2] / name
    target.parent.mkdir(parents=True, exist_ok=True)
    alias.replace(target)
    alias.symlink_to(target)


@pytest.mark.parametrize("sharded", [False, True])
def test_parameter_bound_reads_exact_snapshot_shared_blobs(snapshot, sharded):
    root, status = snapshot
    folder = root / "transformer"
    expected = {"transformer": 24, "text_encoder": 12}
    if sharded:
        write_weights(folder / "weights.safetensors", {
            "one": ("F32", [2], 8), "two": ("F32", [2], 8), "unused": ("F32", [20], 80),
        })
        (folder / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"one": "weights.safetensors", "two": "weights.safetensors"},
        }))
        expected["transformer"] = 8
    aliases = [root / "model_index.json", folder / "weights.safetensors",
               root / "text_encoder" / "weights.safetensors"]
    if sharded:
        aliases.append(folder / "model.safetensors.index.json")
    for number, alias in enumerate(aliases, start=1):
        shared_blob_alias(alias, root.parent.parent.parent, number)

    result = artifact_parameter_lower_bound(status, REVISION, "bfloat16")
    assert result is not None
    assert result["components"] == expected
    assert result["totalBytes"] == sum(expected.values())
    assert result["largestComponentBytes"] == max(expected.values())
    assert result["revision"] == REVISION
    assert impossible_parameter_residency({"accelerator": {"kind": "cuda", "totalBytes": 16}}, result, "none")


def test_parameter_bound_preserves_repository_blob_layout(snapshot):
    root, status = snapshot
    blobs = root.parent.parent / "blobs"
    blobs.mkdir()
    for number, alias in enumerate([
        root / "model_index.json", root / "transformer" / "weights.safetensors",
        root / "text_encoder" / "weights.safetensors",
    ]):
        target = blobs / str(number)
        alias.replace(target)
        alias.symlink_to(target)
    assert artifact_parameter_lower_bound(status, REVISION, "bfloat16")["totalBytes"] == 36


def test_parameter_bound_rejects_linked_component_directory(snapshot):
    root, status = snapshot
    folder = root / "transformer"
    target = root / "other-component"
    folder.rename(target)
    folder.symlink_to(target, target_is_directory=True)
    assert artifact_parameter_lower_bound(status, REVISION, "bfloat16") is None


@pytest.mark.parametrize("member", ["model_index.json", "transformer/model.safetensors.index.json", "transformer/weights.safetensors"])
@pytest.mark.parametrize("target_kind", ["outside", "foreign_repository", "other_snapshot", "malformed_shared_blob"])
def test_parameter_bound_rejects_snapshot_file_link_escapes(snapshot, tmp_path, member, target_kind):
    root, status = snapshot
    (root / "transformer" / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"matrix": "weights.safetensors"},
    }))
    cache_root = root.parent.parent.parent
    targets = {
        "outside": tmp_path / "outside" / "file",
        "foreign_repository": cache_root / "models--foreign--model" / "blobs" / "file",
        "other_snapshot": root.parent / ("b" * 40) / "file",
        "malformed_shared_blob": cache_root / "blobs" / "ff" / ("a" * 64),
    }
    target = targets[target_kind]
    target.parent.mkdir(parents=True, exist_ok=True)
    alias = root / member
    alias.replace(target)
    alias.symlink_to(target)
    assert artifact_parameter_lower_bound(status, REVISION, "bfloat16") is None


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


@pytest.mark.parametrize('quantization', ['none', 'bnb_4bit'])
def test_dense_candidate_publishes_storage_evidence_without_claiming_host_fit(monkeypatch, tmp_path, quantization):
    from modiff import auto_resource as planner

    gib = 1024 ** 3
    candidate = {
        'id': 'native', 'modelType': 'ZImageModularPipeline', 'mode': 'modular_text_to_image',
        'artifactFormat': 'native', 'artifact': planner.Z_IMAGE_REPO,
        'modelRepo': planner.Z_IMAGE_REPO, 'resolvedArtifact': planner.Z_IMAGE_REPO,
        'dtype': 'bfloat16', 'quantizationMode': quantization, 'loadedQuantization': quantization,
        'offloadMode': 'model_cpu', 'installed': True, 'artifactValidation': {'complete': True},
        'artifactResolution': {'resolved': {'repo': planner.Z_IMAGE_REPO, 'revision': REVISION}},
        'proof': {'status': 'declared_safe'}, 'canAutoRun': True, 'requirementsMissing': [],
        'qualityTier': 'native', 'qualityScore': 100, 'rank': 1,
    }
    bound = {'revision': REVISION, 'dtype': 'bfloat16',
             'components': {'transformer': 12 * gib, 'text_encoder': 8 * gib},
             'totalBytes': 20 * gib, 'largestComponentBytes': 12 * gib,
             'proofScope': 'parameter_storage_lower_bound_only'}
    calls = []
    monkeypatch.setattr(planner, '_declared_profile_candidates', lambda *args: [candidate.copy()])
    monkeypatch.setattr(planner, 'artifact_parameter_lower_bound', lambda *args: calls.append(args) or bound.copy())
    hardware = {'accelerator': {'kind': 'cuda', 'totalBytes': 16 * gib, 'freeBytes': 15 * gib},
                'systemMemory': {'totalBytes': 32 * gib, 'availableBytes': 13 * gib},
                'offloadDisk': {'freeBytes': 100 * gib}}
    result = planner.build_auto_resource_plan(
        {'form': {'modelType': 'ZImageModularPipeline', 'mode': 'modular_text_to_image'},
         'hardwareOverride': hardware}, runtime_fingerprint={}, local_models=[], data_dir=tmp_path,
        history={'entries': {}},
    )
    selected = result['selectedCandidate']
    assert selected['canAutoRun']  # Actual owner/cache-dependent admission belongs to the graph plan.
    assert selected['proof']['status'] == 'declared_safe'
    assert selected.get('workingMemoryRequirements') is None
    if quantization == 'none':
        assert selected['parameterStorageLowerBound'] == bound
        assert len(calls) == 1
    else:
        assert 'parameterStorageLowerBound' not in selected
        assert calls == []


@pytest.mark.parametrize('device,mode,expected', [
    ('cuda:0', 'model_cpu', 36), ('cuda:0', 'auto_cpu', 36),
    ('cpu', 'none', 36), ('cpu:0', 'model_cpu', 36), ('cpu:0', 'auto_cpu', 36),
    ('cuda:0', 'none', 36), ('cuda:0', 'group_cpu', 36), ('cuda:0', 'group_disk', 24),
    ('cuda:0', 'sequential_cpu', 0),
])
def test_native_cpu_residency_uses_only_traced_component_placements(snapshot, device, mode, expected):
    _, status = snapshot
    bound = artifact_parameter_lower_bound(status, REVISION, 'bfloat16')
    result = native_cpu_parameter_residency(bound, execution_path='modular-diffusers',
                                           offload_mode=mode, device=device, dtype='bfloat16')
    assert result == ({'systemRamBytes': expected, 'vramBytes': 0} if expected else {})


@pytest.mark.parametrize('change', [
    'missing', 'wrong_revision', 'wrong_dtype', 'wrong_scope', 'wrong_sum',
    'wrong_largest', 'boolean_bytes', 'nonpositive_bytes', 'unsupported_path',
])
def test_unknown_or_invalid_storage_cannot_create_a_cpu_floor(snapshot, change):
    _, status = snapshot
    bound = artifact_parameter_lower_bound(status, REVISION, 'bfloat16')
    path = 'modular-diffusers'
    if change == 'missing':
        bound = None
    elif change == 'unsupported_path':
        path = 'direct-diffusers-video'
    elif change == 'wrong_revision':
        bound['revision'] = 'main'
    elif change == 'wrong_dtype':
        bound['dtype'] = 'float32'
    elif change == 'wrong_scope':
        bound['proofScope'] = 'live_proven'
    elif change == 'wrong_sum':
        bound['totalBytes'] = 37
    elif change == 'wrong_largest':
        bound['largestComponentBytes'] = 36
    elif change == 'boolean_bytes':
        bound['components']['transformer'] = True
    else:
        bound['components']['transformer'] = 0
    assert native_cpu_parameter_residency(bound, execution_path=path, offload_mode='model_cpu',
                                         device='cuda:0', dtype='bfloat16') == {}
