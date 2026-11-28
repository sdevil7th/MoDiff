"""Real tiny header evidence at the ordinary native image Auto boundary.

Only cache installation metadata and finite hardware are controlled. The source
and requested-dtype census uses ordinary safetensors headers, without tensors.
"""

from copy import deepcopy
import json

import pytest

from modiff import artifact_weight_memory as memory
from modiff import auto_resource, workflow_auto_resource
from modiff.workflow_auto_lifecycle import assert_next_owner_capacity
from test_artifact_weight_loading_reserve import write_weights
from test_native_image_auto_placement import (
    GIB, REVISION, _graph, _two_graphs, placement_runtime as _placement_runtime,
)

placement_runtime = _placement_runtime


@pytest.fixture
def loading_evidence(placement_runtime, monkeypatch, tmp_path):
    plan, hardware, bound, _, _ = placement_runtime
    root = tmp_path / "actual-headers" / "models--test--model" / "snapshots" / REVISION
    root.mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({
        "transformer": ["diffusers", "ZImageTransformer2DModel"],
        "text_encoder": ["transformers", "Qwen3Model"],
        "vae": ["diffusers", "AutoencoderKL"],
    }))
    for name, dtype, count in (("transformer", "F32", 17), ("text_encoder", "BF16", 20),
                              ("vae", "BF16", 5)):
        folder = root / name
        folder.mkdir()
        write_weights(folder / "weights.safetensors", {"matrix": (dtype, count)})
    installed = {"complete": True, "snapshots": [{"complete": True, "path": str(root)}]}
    actual_bound = memory.artifact_parameter_lower_bound(installed, REVISION, "bfloat16")
    reserve = {"value": memory.artifact_parameter_conversion_reserve(installed, REVISION, "bfloat16")}
    assert actual_bound["totalBytes"] == 84
    assert reserve["value"]["components"]["transformer"]["serialCpuOverlapBytes"] == 102
    bound.clear()
    bound.update(actual_bound)
    monkeypatch.setattr(auto_resource, "artifact_parameter_conversion_reserve",
                        lambda *args, **kwargs: deepcopy(reserve["value"]))
    # Effective captured-loader semantics are tested separately; this fixture
    # deliberately selects the reviewed serial Diffusers loading contract.
    monkeypatch.setattr(workflow_auto_resource, "diffusers_parallel_loading_enabled", lambda: False, raising=False)
    return plan, hardware, bound, reserve


def exact_owner(graph, *, complete=True, cpu_bytes=0):
    return {"owners": {"models": {
        "cacheKey": workflow_auto_resource.workflow_owner_cache_key(graph, "models"),
        "loadingPhaseReusable": complete, "systemRamBytes": cpu_bytes, "vramBytes": 0,
    }}}


@pytest.mark.parametrize("warm", [False, True])
def test_known_parameter_bound_without_conversion_evidence_cannot_admit_even_a_warm_plan(loading_evidence, warm):
    plan, hardware, _, reserve = loading_evidence
    hardware["systemMemory"]["availableBytes"] = 6 * GIB
    graph = _graph("group_disk")
    before = deepcopy(graph)
    reserve["value"] = None
    result = plan(graph, cache_snapshot=exact_owner(graph) if warm else None)
    assert not result["canAutoRun"]
    assert result["patches"] == []
    assert "complete source and converted-weight headers" in " ".join(result["issues"])
    assert graph == before


@pytest.mark.parametrize("corruption", ["revision", "dtype", "scope", "complete", "missing_component",
                                       "requested_bytes", "numeric_bool", "layout_hash", "source_bytes",
                                       "cast_bytes", "loader_metadata"])
@pytest.mark.parametrize("warm", [False, True])
def test_invalid_conversion_identity_cannot_fall_back_to_requested_storage(loading_evidence, corruption, warm):
    plan, hardware, _, reserve = loading_evidence
    hardware["systemMemory"]["availableBytes"] = 6 * GIB
    graph = _graph("group_disk")
    before = deepcopy(graph)
    record = reserve["value"]
    item = record["components"]["transformer"]
    if corruption == "revision":
        record["revision"] = "a" * 40
    elif corruption == "dtype":
        record["dtype"] = "float32"
    elif corruption == "scope":
        record["proofScope"] = "complete_fit"
    elif corruption == "complete":
        record["complete"] = True
    elif corruption == "missing_component":
        record["components"].pop("vae")
    elif corruption == "requested_bytes":
        item["requestedPayloadBytes"] += 2
    elif corruption == "numeric_bool":
        item["largestSourceShardBytes"] = True
    elif corruption == "layout_hash":
        item["layoutSha256"] = "not-a-sha256"
    elif corruption == "source_bytes":
        item["sourcePayloadBytes"] = 0
    elif corruption == "cast_bytes":
        item["convertedTargetBytes"] = item["requestedPayloadBytes"] + 2
    elif corruption == "loader_metadata":
        item["loaderClass"] = {}
    result = plan(graph, cache_snapshot=exact_owner(graph) if warm else None)
    assert not result["canAutoRun"]
    assert result["patches"] == []
    assert "complete source and converted-weight headers" in " ".join(result["issues"])
    assert graph == before


def test_real_source_cast_overlap_refuses_cold_when_requested_parameter_storage_would_fit(loading_evidence):
    plan, hardware, bound, _ = loading_evidence
    graph = _graph("group_disk")
    hardware["systemMemory"]["availableBytes"] = 4 * GIB + 101
    assert bound["largestComponentBytes"] + 4 * GIB < hardware["systemMemory"]["availableBytes"]
    result = plan(graph)
    assert not result["canAutoRun"]
    assert result["loaders"][0]["parameterResidencyRequirements"]["systemRamBytes"] == 102
    assert result["requirements"]["systemRamBytes"] == 4 * GIB + 102
    assert result["patches"] == []


@pytest.mark.parametrize("complete", [False, True])
def test_cpu_resident_storage_cannot_itself_credit_cold_conversion_overlap(loading_evidence, complete):
    plan, hardware, bound, _ = loading_evidence
    graph = _graph("model_cpu")
    hardware["systemMemory"]["availableBytes"] = 4 * GIB + 60
    result = plan(graph, cache_snapshot=exact_owner(graph, complete=complete, cpu_bytes=bound["totalBytes"]))
    owner = result["loaders"][0]
    assert owner["parameterResidencyRequirements"]["systemRamBytes"] == 84
    assert owner["parameterConversionRequirements"]["systemRamBytes"] == 152
    assert owner["parameterConversionAdditionalBytes"] == 68
    assert result["requirements"]["systemRamBytes"] == 4 * GIB + (0 if complete else 68)
    assert result["canAutoRun"] is complete
    assert result["patches"] == []


def test_complete_old_owner_policy_cannot_waive_new_loading_inputs(loading_evidence):
    plan, hardware, bound, _ = loading_evidence
    graph = _graph("model_cpu")
    cache = exact_owner(graph, cpu_bytes=bound["totalBytes"])
    graph["nodes"]["models"]["params"]["vae_tiling"]["value"] = True
    hardware["systemMemory"]["availableBytes"] = 4 * GIB + 101
    result = plan(graph, cache_snapshot=cache)
    assert not result["canAutoRun"]
    assert result["reusedOwnerIds"] == []
    assert result["patches"] == []


@pytest.mark.parametrize("short_pool", [None, "systemRamBytes", "vramBytes"])
def test_scheduled_cuda_loading_restores_both_cold_pools_and_rechecks_actual_free_memory(loading_evidence, short_pool):
    plan, hardware, _, _ = loading_evidence
    graph = _two_graphs()
    for node_id in ("models", "second_models"):
        graph["nodes"][node_id]["params"].update(
            offload_mode={"value": "group_disk"}, weight_loading_mode={"value": "cuda"},
        )
    hardware["systemMemory"]["availableBytes"] = 4 * GIB + 102
    hardware["accelerator"]["freeBytes"] = 2 * GIB + 40
    cache = {"owners": {node_id: {
        "cacheKey": workflow_auto_resource.workflow_owner_cache_key(graph, node_id),
        "loadingPhaseReusable": True, "systemRamBytes": 0, "vramBytes": 0,
    } for node_id in ("models", "second_models")}}
    result = plan(graph, cache_snapshot=cache)
    assert result["canAutoRun"], result["issues"]
    assert result["schedule"] is not None
    assert result["reusedOwnerIds"] == []
    assert result["requirements"]["systemRamBytes"] == 4 * GIB + 102
    assert result["requirements"]["vramBytes"] == 2 * GIB + 40
    owner = result["schedule"]["owners"][1]
    assert owner["parameterResidencyRequirements"]["systemRamBytes"] == 102
    assert owner["parameterLoadingRequirements"] == {"vramBytes": 40}
    after_release = deepcopy(hardware)
    if short_pool is not None:
        if short_pool == "systemRamBytes":
            after_release["systemMemory"]["availableBytes"] -= 1
        else:
            after_release["accelerator"]["freeBytes"] -= 1
        with pytest.raises(ValueError, match="actual free memory after the release"):
            assert_next_owner_capacity(object(), owner, after_release)
    else:
        assert_next_owner_capacity(object(), owner, after_release)
