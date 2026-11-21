"""Pure declared loading contracts; no tensor import or live memory proof."""

import copy
import importlib
import json
import struct
import sys

import pytest

from modiff import artifact_weight_memory as memory
from modiff import native_weight_loading as contract


REVISION = "a" * 40
CLASSES = {"transformer": ["diffusers", "ZImageTransformer2DModel"],
           "vae": ["diffusers", "AutoencoderKL"], "text_encoder": ["transformers", "Qwen3Model"]}


def weights(path, tensors):
    offset, header = 0, {}
    for name, (dtype, count) in tensors.items():
        size = count * (4 if dtype == "F32" else 2)
        header[name] = {"shape": [count], "dtype": dtype, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))


@pytest.fixture
def census(tmp_path):
    root = tmp_path / "hub" / "models--test--contract" / "snapshots" / REVISION
    root.mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps(CLASSES))
    for name, (dtype, count) in {"transformer": ("F32", 17), "text_encoder": ("BF16", 20), "vae": ("BF16", 5)}.items():
        folder = root / name
        folder.mkdir()
        weights(folder / "weights.safetensors", {"matrix": (dtype, count)})
    status = {"complete": True, "snapshots": [{"complete": True, "path": str(root)}]}

    def inspect(dtype="bfloat16"):
        return (memory.artifact_parameter_conversion_reserve(status, REVISION, dtype),
                memory.artifact_parameter_lower_bound(status, REVISION, dtype))

    return root, inspect


@pytest.mark.parametrize("field,value", [("loaderLibrary", {}), ("loaderLibrary", []),
                                         ("loaderClass", {}), ("loaderClass", [])])
def test_unhashable_loader_metadata_fails_closed(census, field, value):
    _, inspect = census
    reserve, bound = inspect()
    reserve["components"]["transformer"][field] = value
    assert contract.declared_cuda_weight_loaders(reserve, bound) is False


def test_opaque_string_metadata_is_rejected_without_executing_its_methods(census):
    class OpaqueString(str):
        def __hash__(self):
            raise AssertionError("Unreviewed metadata hash executed")

        def __eq__(self, other):
            raise AssertionError("Unreviewed metadata comparison executed")

    _, inspect = census
    reserve, bound = inspect()
    reserve["components"]["transformer"]["loaderClass"] = OpaqueString("ZImageTransformer2DModel")
    assert contract.declared_cuda_weight_loaders(reserve, bound) is False
    with pytest.raises(ValueError, match="must be cpu or cuda"):
        contract.normalize_weight_loading_mode(OpaqueString("cuda"))


def test_exact_component_roles_and_bfloat16_metadata_are_declared_without_model_imports(census):
    _, inspect = census
    reserve, bound = inspect()
    assert contract.declared_cuda_weight_loaders(reserve, bound) is True
    assert reserve["complete"] is False  # Declaration is not complete memory-fit proof.
    assert bound["components"] == {"transformer": 34, "vae": 10, "text_encoder": 40}


@pytest.mark.parametrize("component,pair", [
    ("transformer", ["diffusers", "AutoencoderKL"]),
    ("vae", ["diffusers", "ZImageTransformer2DModel"]),
    ("text_encoder", ["diffusers", "AutoencoderKL"]),
    ("transformer", ["diffusers", "OtherTransformer"]),
    ("text_encoder", ["transformers", "AutoModel"]),
    ("vae", ["remote-model-code", "AutoencoderKL"]),
])
def test_reviewed_namespaces_or_classes_do_not_authorize_the_wrong_component(census, component, pair):
    root, inspect = census
    metadata = copy.deepcopy(CLASSES)
    metadata[component] = pair
    (root / "model_index.json").write_text(json.dumps(metadata))
    reserve, bound = inspect()
    assert reserve and bound
    assert contract.declared_cuda_weight_loaders(reserve, bound) is False


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_cuda_declaration_is_limited_to_reviewed_uniform_requested_dtype(census, dtype):
    _, inspect = census
    reserve, bound = inspect(dtype)
    assert reserve and bound
    assert contract.declared_cuda_weight_loaders(reserve, bound) is False


@pytest.mark.parametrize("change", ["missing", "empty", "additional_role", "missing_record", "missing_bound", "malformed"])
def test_incomplete_or_additional_component_metadata_does_not_expand_declared_scope(census, change):
    _, inspect = census
    reserve, bound = inspect()
    if change == "missing":
        reserve = None
    elif change == "empty":
        reserve["components"] = {}
        bound["components"] = {}
    elif change == "additional_role":
        reserve["components"]["controlnet"] = reserve["components"]["transformer"].copy()
        bound["components"]["controlnet"] = 34
    elif change == "missing_record":
        del reserve["components"]["text_encoder"]
    elif change == "missing_bound":
        del bound["components"]["text_encoder"]
    else:
        reserve["components"]["transformer"] = []
    assert contract.declared_cuda_weight_loaders(reserve, bound) is False


def test_a_declared_class_is_separate_from_full_reserve_identity_validation(census):
    _, inspect = census
    reserve, bound = inspect()
    reserve["revision"] = "b" * 40
    assert contract.declared_cuda_weight_loaders(reserve, bound)  # This predicate checks only declarations.
    assert memory.native_parameter_conversion_reserve(
        reserve, lower_bound=bound, execution_path="modular-diffusers", offload_mode="group_disk",
        device="cuda:0", dtype="bfloat16", weight_loading_mode="cuda", parallel_loading=False,
    ) == {}


@pytest.mark.parametrize("value", ["cpu", "cuda"])
def test_only_supported_loading_enum_is_normalized(value):
    assert contract.normalize_weight_loading_mode(value) == value
    assert contract.normalize_weight_loading_mode() == "cpu"


@pytest.mark.parametrize("value", [None, False, True, 0, 1, {}, [], "", "CPU", "CUDA", "auto", " cuda", "cpu "])
def test_unknown_or_unhashable_loading_mode_has_an_actionable_error(value):
    with pytest.raises(ValueError, match="must be cpu or cuda"):
        contract.normalize_weight_loading_mode(value)


@pytest.mark.parametrize("value", [{}, [], None, False, 0])
def test_artifact_reserve_enum_guard_precedes_unhashable_membership(census, value):
    _, inspect = census
    reserve, bound = inspect()
    assert memory.native_parameter_conversion_reserve(
        reserve, lower_bound=bound, execution_path="modular-diffusers", offload_mode="group_disk",
        device="cuda:0", dtype="bfloat16", weight_loading_mode=value, parallel_loading=False,
    ) == {}


def lora_reserve(census, targets, *, parallel=False, dtype="bfloat16", mutate=None):
    _, inspect = census
    reserve, bound = inspect(dtype)
    if mutate:
        mutate(reserve, bound)
    return contract.native_disk_lora_conversion_reserve(
        reserve, lower_bound=bound, targets=targets, dtype=dtype, device="cuda:0",
        parallel_loading=parallel,
    )


@pytest.mark.parametrize("targets", [("transformer", "text_encoder"), ("text_encoder", "transformer")])
def test_deferred_targets_keep_other_requested_targets_beside_current_cast(census, targets):
    _, inspect = census
    reserve, bound = inspect()
    generic = memory.native_parameter_conversion_reserve(
        reserve, lower_bound=bound, execution_path="modular-diffusers", offload_mode="group_disk",
        device="cuda:0", dtype="bfloat16", parallel_loading=False,
    )
    assert generic["systemRamBytes"] == 102
    assert memory.native_disk_lora_loading_floor(bound, targets=targets, dtype="bfloat16")["systemRamBytes"] == 74
    # Taking max(102,74) loses the other retained target's 40 bytes.
    assert lora_reserve(census, targets) == {"systemRamBytes": 142, "vramBytes": 0}
    assert reserve["complete"] is False


def test_non_target_component_loading_first_is_not_added_to_later_staged_targets(census):
    root, _ = census
    weights(root / "vae" / "weights.safetensors", {"matrix": ("F32", 100)})
    # VAE alone has 400 source+200 cast; it disk-offloads before target loading.
    assert lora_reserve(census, ("transformer", "text_encoder"))["systemRamBytes"] == 600


@pytest.mark.parametrize("target", ["transformer", "text_encoder", "vae"])
def test_single_deferred_target_keeps_the_other_component_peak(census, target):
    assert lora_reserve(census, (target,)) == {"systemRamBytes": 102, "vramBytes": 0}


def test_same_dtype_deferred_targets_do_not_receive_a_second_storage_copy(census):
    root, _ = census
    weights(root / "transformer" / "weights.safetensors", {"matrix": ("BF16", 17)})
    assert lora_reserve(census, ("transformer", "text_encoder")) == {"systemRamBytes": 74, "vramBytes": 0}


def test_serial_and_parallel_staged_target_envelopes_use_actual_header_shards(census):
    root, _ = census
    folder = root / "transformer"
    (folder / "weights.safetensors").unlink()
    weights(folder / "a.safetensors", {"first": ("F32", 5)})
    weights(folder / "z.safetensors", {"second": ("F32", 12)})
    (folder / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"second": "z.safetensors", "first": "a.safetensors"},
    }))
    assert lora_reserve(census, ("transformer", "text_encoder"), parallel=False)["systemRamBytes"] == 122
    assert lora_reserve(census, ("transformer", "text_encoder"), parallel=True)["systemRamBytes"] == 142


@pytest.mark.parametrize("targets", [(), ("missing",), ("transformer", "transformer"),
                                    ["transformer"], None, ({},), ([],)])
def test_missing_invalid_or_unhashable_targets_fail_closed(census, targets):
    assert lora_reserve(census, targets) == {}


@pytest.mark.parametrize("change", ["revision", "dtype", "scope", "requested", "missing_component", "missing_header"])
def test_staged_conversion_cannot_inherit_mismatched_or_incomplete_evidence(census, change):
    if change == "missing_header":
        root, _ = census
        (root / "transformer" / "weights.safetensors").unlink()
        assert lora_reserve(census, ("transformer", "text_encoder")) == {}
        return

    def mutate(reserve, bound):
        if change == "revision":
            reserve["revision"] = "b" * 40
        elif change == "dtype":
            reserve["dtype"] = "float32"
        elif change == "scope":
            reserve["proofScope"] = "complete_working_demand"
        elif change == "requested":
            reserve["components"]["transformer"]["requestedPayloadBytes"] += 1
        else:
            del reserve["components"]["text_encoder"]

    assert lora_reserve(census, ("transformer", "text_encoder"), mutate=mutate) == {}


@pytest.mark.parametrize("parallel", [None, 0, "false", {}])
def test_staged_conversion_unknown_flag_does_not_borrow_serial_narrowing(census, parallel):
    assert lora_reserve(census, ("transformer",), parallel=parallel) == {}


def test_pure_contracts_import_no_heavy_packages_or_execute_models(census, monkeypatch):
    import builtins

    original = builtins.__import__
    heavy = {"torch", "diffusers", "transformers", "accelerate", "peft", "safetensors"}

    def checked(name, *args, **kwargs):
        assert name.split(".", 1)[0] not in heavy
        return original(name, *args, **kwargs)

    before = set(sys.modules)
    monkeypatch.setattr(builtins, "__import__", checked)
    importlib.reload(contract)
    _, inspect = census
    reserve, bound = inspect()
    assert contract.declared_cuda_weight_loaders(reserve, bound)
    assert lora_reserve(census, ("transformer", "text_encoder"))["systemRamBytes"] == 142
    assert not heavy & {name.split(".", 1)[0] for name in set(sys.modules) - before}
