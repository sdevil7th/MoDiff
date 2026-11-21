"""Header-only loading overlap evidence, never live allocation or fit proof."""

import copy
import importlib
import json
import struct
import sys

import pytest

from modiff import artifact_weight_memory as memory


REVISION = "a" * 40


def write_weights(path, tensors):
    header, offset = {}, 0
    sizes = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8, "I32": 4}
    for name, (dtype, count) in tensors.items():
        size = sizes[dtype] * count
        header[name] = {"dtype": dtype, "shape": [count], "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "hub" / "models--test--model" / "snapshots" / REVISION
    folder = root / "transformer"
    folder.mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({"transformer": ["diffusers", "Transformer"]}))
    # Intentionally insert the later shard first: upstream sorts filenames.
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "second": "z.safetensors", "first": "a.safetensors",
    }}))
    write_weights(folder / "a.safetensors", {"first": ("F32", 8), "unused": ("F32", 20)})
    write_weights(folder / "z.safetensors", {"second": ("F32", 12)})
    status = {"complete": True, "snapshots": [{"complete": True, "path": str(root)}]}
    return root, status


def census(snapshot, dtype="bfloat16"):
    _, status = snapshot
    function = getattr(memory, "artifact_parameter_conversion_reserve", lambda *args: None)
    return function(status, REVISION, dtype)


@pytest.mark.parametrize("document", [None, [], "not an object", 1, True])
@pytest.mark.parametrize("reader", [
    memory.artifact_parameter_lower_bound, memory.artifact_parameter_conversion_reserve,
])
def test_non_object_shard_index_is_unknown_storage(snapshot, document, reader):
    root, status = snapshot
    (root / "transformer" / "model.safetensors.index.json").write_text(json.dumps(document))
    assert reader(status, REVISION, "bfloat16") is None


def test_cold_cast_and_unindexed_source_need_separate_evidence(snapshot):
    _, status = snapshot
    before = memory.artifact_parameter_lower_bound(status, REVISION, "bfloat16")
    assert before["components"] == {"transformer": 40}
    reserve = census(snapshot)
    assert reserve is not None, "Requested storage alone omits source/cast cold overlap"
    component = reserve["components"]["transformer"]
    assert component["sourcePayloadBytes"] == 160
    assert component["requestedPayloadBytes"] == 40
    assert component["convertedTargetBytes"] == 40
    assert component["largestSourceShardBytes"] == 112
    assert component["largestConvertedTensorBytes"] == 24
    assert component["serialCpuOverlapBytes"] == 128  # source a + selected converted a
    assert component["serialCudaOverlapBytes"] == 128
    assert component["conservativeCpuOverlapBytes"] == 200
    assert component["conservativeCudaOverlapBytes"] == 200
    assert reserve["proofScope"] == "source_requested_loading_overlap_only"
    assert reserve["complete"] is False
    assert memory.artifact_parameter_lower_bound(status, REVISION, "bfloat16") == before


def native(snapshot, *, dtype="bfloat16", reserve=None, **kwargs):
    _, status = snapshot
    return memory.native_parameter_conversion_reserve(
        census(snapshot, dtype) if reserve is None else reserve,
        lower_bound=memory.artifact_parameter_lower_bound(status, REVISION, dtype),
        execution_path="modular-diffusers", offload_mode=kwargs.pop("offload_mode", "group_disk"),
        device=kwargs.pop("device", "cuda:0"), dtype=dtype, **kwargs,
    )


def unsharded(snapshot, tensors):
    root, _ = snapshot
    folder = root / "transformer"
    for path in folder.iterdir():
        path.unlink()
    write_weights(folder / "weights.safetensors", tensors)


def test_serial_shard_order_is_sorted_and_prefix_requested_destinations_are_retained(snapshot):
    root, _ = snapshot
    folder = root / "transformer"
    # Ascending file order gives 16+48+24=88, rather than the reversed 24+32+16=72.
    write_weights(folder / "a.safetensors", {"first": ("F32", 8)})
    result = census(snapshot)["components"]["transformer"]
    assert result["serialCpuOverlapBytes"] == 88
    assert result["serialCudaOverlapBytes"] == 72
    assert result["conservativeCpuOverlapBytes"] == 120
    assert native(snapshot, parallel_loading=False)["systemRamBytes"] == 88
    assert native(snapshot, parallel_loading=True)["systemRamBytes"] == 120


@pytest.mark.parametrize("stored,dtype", [("BF16", "bfloat16"), ("F32", "float32"), ("F16", "float16"), ("F64", "float64")])
def test_same_dtype_selected_storage_aliases_without_duplication(snapshot, stored, dtype):
    unsharded(snapshot, {"matrix": (stored, 8)})
    item = census(snapshot, dtype)["components"]["transformer"]
    size = memory._DTYPE_BYTES[dtype] * 8
    assert item["sourcePayloadBytes"] == item["requestedPayloadBytes"] == size
    assert item["convertedTargetBytes"] == item["largestConvertedTensorBytes"] == 0
    assert item["serialCpuOverlapBytes"] == item["conservativeCpuOverlapBytes"] == size
    assert native(snapshot, dtype=dtype, parallel_loading=False)["systemRamBytes"] == size


def test_same_dtype_unused_source_still_needs_cold_overlap(snapshot):
    item = census(snapshot, "float32")["components"]["transformer"]
    assert item["requestedPayloadBytes"] == 80
    assert item["sourcePayloadBytes"] == 160
    assert item["convertedTargetBytes"] == 0
    assert item["serialCpuOverlapBytes"] == 112
    assert item["conservativeCpuOverlapBytes"] == 160


def test_mixed_source_dtype_and_same_byte_width_cast_are_distinct(snapshot):
    unsharded(snapshot, {"same": ("BF16", 2), "f32": ("F32", 2), "f16": ("F16", 2),
                         "integer": ("I32", 2), "f64": ("F64", 2)})
    item = census(snapshot)["components"]["transformer"]
    assert item["sourcePayloadBytes"] == 40
    assert item["requestedPayloadBytes"] == 24
    assert item["convertedTargetBytes"] == 12  # F16 -> BF16 copies despite equal width.
    assert item["largestConvertedTensorBytes"] == 4
    assert item["serialCpuOverlapBytes"] == 52


def test_cuda_serial_keeps_every_current_shard_cast_and_disk_serialization_copy(snapshot):
    root, _ = snapshot
    folder = root / "transformer"
    write_weights(folder / "a.safetensors", {"first": ("F32", 8)})
    assert native(snapshot, weight_loading_mode="cuda", parallel_loading=False) == {"systemRamBytes": 72, "vramBytes": 0}
    assert native(snapshot, weight_loading_mode="cuda", parallel_loading=True) == {"systemRamBytes": 120, "vramBytes": 0}
    # Two casts in one shard count together; largest individual cast is diagnostic only.
    unsharded(snapshot, {"a": ("F32", 10), "b": ("F32", 10)})
    item = census(snapshot)["components"]["transformer"]
    assert item["largestConvertedTensorBytes"] == 20
    assert native(snapshot, weight_loading_mode="cuda", parallel_loading=False)["systemRamBytes"] == 120


def test_cuda_disk_serialization_can_dominate_source_cast_phase(snapshot):
    root, _ = snapshot
    folder = root / "transformer"
    write_weights(folder / "a.safetensors", {"first": ("F32", 8)})
    write_weights(folder / "z.safetensors", {"second": ("F32", 12)})
    # F32 -> F64: second source+cast=144, total requested disk copy=160.
    item = census(snapshot, "float64")["components"]["transformer"]
    assert item["serialCudaOverlapBytes"] == 144
    assert native(snapshot, dtype="float64", weight_loading_mode="cuda", parallel_loading=False)["systemRamBytes"] == 160


@pytest.mark.parametrize("library", ["transformers", "unknown-official-loader"])
def test_non_diffusers_library_never_borrows_diffusers_serial_order(snapshot, library):
    root, _ = snapshot
    (root / "model_index.json").write_text(json.dumps({"transformer": [library, "Encoder"]}))
    assert native(snapshot, parallel_loading=False)["systemRamBytes"] == 200
    assert native(snapshot, weight_loading_mode="cuda", parallel_loading=False)["systemRamBytes"] == 200


@pytest.mark.parametrize("device,mode", [
    ("cpu", "none"), ("cpu:0", "model_cpu"), ("cuda:0", "none"),
    ("cuda:0", "model_cpu"), ("cuda:0", "auto_cpu"), ("cuda:0", "group_cpu"),
])
def test_retained_cpu_components_do_not_assume_model_index_iteration_order(snapshot, device, mode):
    root, _ = snapshot
    (root / "model_index.json").write_text(json.dumps({
        "other": ["diffusers", "Encoder"], "transformer": ["diffusers", "Transformer"],
    }))
    folder = root / "other"
    folder.mkdir()
    write_weights(folder / "weights.safetensors", {"matrix": ("BF16", 50)})
    # all requested=140 plus largest conversion/unused-source excess=128-40=88.
    assert native(snapshot, device=device, offload_mode=mode, parallel_loading=False)["systemRamBytes"] == 228
    assert native(snapshot, parallel_loading=False)["systemRamBytes"] == 128


@pytest.mark.parametrize("changes", [
    {"weight_loading_mode": "unknown"}, {"weight_loading_mode": "cuda", "device": "cpu"},
    {"weight_loading_mode": "cuda", "device": "cuda:1"},
    {"weight_loading_mode": "cuda", "offload_mode": "none"},
    {"offload_mode": "sequential_cpu"}, {"parallel_loading": None},
    {"parallel_loading": 0}, {"parallel_loading": "false"},
])
def test_unsupported_or_unknown_loading_policy_does_not_grant_a_smaller_reserve(snapshot, changes):
    assert native(snapshot, **changes) == {}


@pytest.mark.parametrize("change", [
    "revision", "dtype", "scope", "complete", "missing_component", "extra_component", "requested",
    "boolean", "negative", "overflow", "parallel_sum", "serial_less_than_source", "layout",
    "missing_library", "converted_exceeds_requested",
])
def test_native_admission_requires_matching_storage_evidence_and_valid_companion(snapshot, change):
    reserve = copy.deepcopy(census(snapshot))
    component = reserve["components"]["transformer"]
    if change in {"revision", "dtype", "scope", "complete"}:
        key = "proofScope" if change == "scope" else change
        reserve[key] = {"revision": "b" * 40, "dtype": "float32", "scope": "live_proven", "complete": True}[change]
    elif change == "missing_component":
        reserve["components"] = {}
    elif change == "extra_component":
        reserve["components"]["extra"] = component.copy()
    else:
        key, value = {
            "requested": ("requestedPayloadBytes", 41), "boolean": ("sourcePayloadBytes", True),
            "negative": ("convertedTargetBytes", -1), "overflow": ("sourcePayloadBytes", 2**64),
            "parallel_sum": ("conservativeCpuOverlapBytes", 199),
            "serial_less_than_source": ("serialCudaOverlapBytes", 111),
            "layout": ("layoutSha256", "mutable"), "missing_library": ("loaderLibrary", ""),
            "converted_exceeds_requested": ("convertedTargetBytes", 41),
        }[change]
        component[key] = value
    assert native(snapshot, reserve=reserve, parallel_loading=False) == {}


@pytest.mark.parametrize("revision,dtype,complete", [
    ("b" * 40, "bfloat16", True), ("main", "bfloat16", True),
    (REVISION, "int4", True), (REVISION, "bfloat16", False),
])
def test_missing_or_unsupported_snapshot_never_supplies_conversion_evidence(snapshot, revision, dtype, complete):
    _, status = snapshot
    status = copy.deepcopy(status)
    status["complete"] = complete
    assert memory.artifact_parameter_conversion_reserve(status, revision, dtype) is None


@pytest.mark.parametrize("filename", ["../a.safetensors", "..\\a.safetensors", "/a.safetensors", "C:a.safetensors", "weights.bin"])
def test_conversion_index_retains_exact_managed_cache_boundary(snapshot, filename):
    root, _ = snapshot
    (root / "transformer" / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"first": filename}}))
    assert census(snapshot) is None


@pytest.mark.parametrize("change", ["oversized", "short", "shape", "offset", "overlap", "gap", "unknown_dtype", "duplicate", "missing_indexed", "unused_invalid"])
def test_all_source_headers_must_be_finite_and_valid_including_unused_tensors(snapshot, change):
    root, _ = snapshot
    file = root / "transformer" / "a.safetensors"
    raw = file.read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + size])
    payload = raw[8 + size:]
    if change == "oversized":
        file.write_bytes(struct.pack("<Q", 16 * 1024 * 1024 + 1))
    elif change == "short":
        file.write_bytes(b"short")
    else:
        if change == "shape":
            header["first"]["shape"] = [True]
        elif change == "offset":
            header["first"]["data_offsets"][1] = 999
        elif change == "overlap":
            header["unused"]["data_offsets"] = [0, 80]
        elif change == "gap":
            header["unused"]["data_offsets"] = [33, 113]
            payload += b"x"
        elif change == "unknown_dtype":
            header["first"]["dtype"] = "F8_E4M3"
        elif change == "missing_indexed":
            del header["first"]
        elif change == "unused_invalid":
            header["unused"]["shape"] = [-20]
        data = json.dumps(header).encode()
        if change == "duplicate":
            data = b'{"first":{},' + data[1:]
        file.write_bytes(struct.pack("<Q", len(data)) + data + payload)
    assert census(snapshot) is None


def test_index_json_duplicates_and_multiple_layouts_are_unknown(snapshot):
    root, _ = snapshot
    folder = root / "transformer"
    (folder / "model.safetensors.index.json").write_text('{"weight_map":{"first":"a.safetensors","first":"z.safetensors"}}')
    assert census(snapshot) is None
    (folder / "other.safetensors.index.json").write_text('{"weight_map":{}}')
    assert census(snapshot) is None


@pytest.mark.parametrize("target_kind", ["outside", "foreign_repository", "other_snapshot"])
def test_conversion_header_link_escapes_remain_unknown(snapshot, tmp_path, target_kind):
    root, _ = snapshot
    cache = root.parent.parent.parent
    target = {"outside": tmp_path / "outside" / "file",
              "foreign_repository": cache / "models--foreign--model" / "blobs" / "file",
              "other_snapshot": root.parent / ("b" * 40) / "file"}[target_kind]
    target.parent.mkdir(parents=True)
    alias = root / "transformer" / "a.safetensors"
    alias.replace(target)
    alias.symlink_to(target)
    assert census(snapshot) is None


def test_conversion_metadata_uses_supported_shared_blob_alias(snapshot):
    root, _ = snapshot
    target = root.parent.parent.parent / "blobs" / "00" / ("0" * 64)
    target.parent.mkdir(parents=True)
    alias = root / "transformer" / "a.safetensors"
    alias.replace(target)
    alias.symlink_to(target)
    assert census(snapshot)["components"]["transformer"]["serialCpuOverlapBytes"] == 128


def test_layout_identity_changes_with_headers_and_selected_index_mapping(snapshot):
    root, _ = snapshot
    before = census(snapshot)["components"]["transformer"]["layoutSha256"]
    (root / "transformer" / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"first": "a.safetensors", "unused": "a.safetensors", "second": "z.safetensors"},
    }))
    after = census(snapshot)["components"]["transformer"]["layoutSha256"]
    assert after != before


@pytest.mark.parametrize("kind", ["components", "shards", "tensor_count", "snapshots"])
def test_loading_metadata_inventories_remain_bounded(snapshot, kind):
    root, status = snapshot
    if kind == "components":
        (root / "model_index.json").write_text(json.dumps({f"component{i}": ["diffusers", "Model"] for i in range(129)}))
    elif kind == "shards":
        (root / "transformer" / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {f"tensor{i}": f"file{i}.safetensors" for i in range(1025)},
        }))
    elif kind == "tensor_count":
        raw = json.dumps({f"tensor{i}": {} for i in range(100001)}).encode()
        (root / "transformer" / "a.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    else:
        status["snapshots"] *= 1025
    assert census(snapshot) is None


@pytest.mark.parametrize("status", [None, [], {"complete": True, "snapshots": {}},
                                    {"complete": True, "snapshots": [{"complete": True, "path": {}}]},
                                    {"complete": 1, "snapshots": []}])
def test_malformed_cache_status_is_unknown_without_execution(status):
    assert memory.artifact_parameter_conversion_reserve(status, REVISION, "bfloat16") is None


@pytest.mark.parametrize("revision,dtype", [(None, "bfloat16"), ([], "bfloat16"), (REVISION, []), (REVISION, True)])
def test_malformed_revision_or_dtype_is_unknown(snapshot, revision, dtype):
    _, status = snapshot
    assert memory.artifact_parameter_conversion_reserve(status, revision, dtype) is None


def test_linked_component_directory_is_not_loading_proof(snapshot):
    root, _ = snapshot
    original = root / "transformer"
    target = root / "other-component"
    original.rename(target)
    original.symlink_to(target, target_is_directory=True)
    assert census(snapshot) is None


def test_header_metadata_must_match_safetensors_string_contract(snapshot):
    root, _ = snapshot
    file = root / "transformer" / "a.safetensors"
    raw = file.read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + size])
    header["__metadata__"] = {"invalid": {}}
    data = json.dumps(header).encode()
    file.write_bytes(struct.pack("<Q", len(data)) + data + raw[8 + size:])
    assert census(snapshot) is None


def test_census_and_native_helper_import_no_tensor_or_model_packages(snapshot, monkeypatch):
    import builtins

    original = builtins.__import__
    calls = []

    def checked(name, *args, **kwargs):
        calls.append(name)
        assert name.split(".", 1)[0] not in {"torch", "diffusers", "transformers", "accelerate", "peft", "safetensors"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked)
    before = set(sys.modules)
    importlib.reload(memory)
    assert native(snapshot, parallel_loading=False)["systemRamBytes"] == 128
    assert not {name.split(".", 1)[0] for name in set(sys.modules) - before} & {"torch", "diffusers", "transformers", "peft", "accelerate", "safetensors"}
    assert calls
