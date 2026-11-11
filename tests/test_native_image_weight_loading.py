"""Cold loading phases use ordinary graph controls, never inferred free bytes.

These regressions exercise both existing planners and the real image profile.
Header overlap and hardware are controlled evidence; no model is loaded here.
"""

from copy import deepcopy

import pytest

from modiff import auto_resource, workflow_auto_resource
from test_native_image_auto_placement import (
    GIB, PARAMETERS, _graph, placement_runtime as _placement_runtime,
)

placement_runtime = _placement_runtime
CPU_CONVERSION = 19_947_373_696
CUDA_CONVERSION = 14_960_538_624


@pytest.fixture
def loading_runtime(placement_runtime, monkeypatch):
    evidence = {"controlledHeaderOverlap": True, "dtype": "bfloat16", "components": {
        "transformer": {"loaderLibrary": "diffusers", "loaderClass": "ZImageTransformer2DModel"},
        "vae": {"loaderLibrary": "diffusers", "loaderClass": "AutoencoderKL"},
        "text_encoder": {"loaderLibrary": "transformers", "loaderClass": "Qwen3Model"},
    }}
    monkeypatch.setenv("HF_ENABLE_PARALLEL_LOADING", "0")
    monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", "1")
    monkeypatch.setattr(auto_resource, "artifact_parameter_conversion_reserve",
                        lambda *args, **kwargs: deepcopy(evidence), raising=False)

    def reserve(value, *, lower_bound, weight_loading_mode="cpu", **kwargs):
        if value != evidence or lower_bound != PARAMETERS:
            return {}
        return {"systemRamBytes": CUDA_CONVERSION if weight_loading_mode == "cuda" else CPU_CONVERSION,
                "vramBytes": 0}

    monkeypatch.setattr(workflow_auto_resource, "native_parameter_conversion_reserve", reserve, raising=False)
    return placement_runtime


def test_original_cpu_disk_admission_cannot_omit_f32_conversion_overlap(loading_runtime):
    plan, hardware, _, _, _ = loading_runtime
    graph = _graph("group_disk")
    before = deepcopy(graph)
    assert PARAMETERS["largestComponentBytes"] + 4 * GIB < hardware["systemMemory"]["availableBytes"]
    assert CPU_CONVERSION + 4 * GIB > hardware["systemMemory"]["availableBytes"]
    result = plan(graph)
    assert not result["canAutoRun"], "Cold CPU source and converted storage exceed actual free RAM."
    assert result["patches"] == []
    assert graph == before


def test_new_generic_control_can_select_admitted_cuda_loading_without_creative_changes(loading_runtime):
    plan, hardware, _, _, _ = loading_runtime
    graph = _graph("group_disk")
    graph["nodes"]["models"]["params"]["weight_loading_mode"] = {"value": "cpu"}
    before = deepcopy(graph)
    assert CUDA_CONVERSION + 4 * GIB < hardware["systemMemory"]["availableBytes"]
    assert PARAMETERS["largestComponentBytes"] + 2 * GIB < hardware["accelerator"]["freeBytes"]
    result = plan(graph)
    assert result["canAutoRun"], result["issues"]
    assert result["loaders"][0]["settings"]["weight_loading_mode"] == "cuda"
    assert result["requirements"]["systemRamBytes"] == CUDA_CONVERSION + 4 * GIB
    assert result["requirements"]["vramBytes"] == PARAMETERS["largestComponentBytes"] + 2 * GIB
    assert graph == before
    assert result["patches"] == [{"nodeId": "models", "field": "weight_loading_mode", "value": "cuda"}]


@pytest.mark.parametrize("gpu_free", [0, 2 * GIB, PARAMETERS["largestComponentBytes"] + 2 * GIB - 1])
def test_cuda_loading_cannot_use_the_group_inference_floor_as_its_cold_gpu_budget(loading_runtime, gpu_free):
    plan, hardware, _, _, _ = loading_runtime
    hardware["accelerator"]["freeBytes"] = gpu_free
    graph = _graph("group_disk")
    graph["nodes"]["models"]["params"]["weight_loading_mode"] = {"value": "cuda"}
    before = deepcopy(graph)
    result = plan(graph)
    assert not result["canAutoRun"]
    assert result["patches"] == []
    assert graph == before


def test_complete_warm_disk_owner_waives_cold_gpu_staging_but_preserves_real_headroom(loading_runtime):
    plan, hardware, _, _, _ = loading_runtime
    graph = _graph("group_disk")
    graph["nodes"]["models"]["params"]["weight_loading_mode"] = {"value": "cuda"}
    hardware["systemMemory"]["availableBytes"] = 6 * GIB
    hardware["accelerator"]["freeBytes"] = 3 * GIB
    cache = {"owners": {"models": {
        "cacheKey": workflow_auto_resource.workflow_owner_cache_key(graph, "models"),
        "loadingPhaseReusable": True, "systemRamBytes": 0, "vramBytes": 0,
    }}}
    result = plan(graph, cache_snapshot=cache)
    assert result["canAutoRun"], result["issues"]
    assert result["reusedOwnerIds"] == ["models"]
    assert result["requirements"]["systemRamBytes"] == 4 * GIB
    assert result["requirements"]["vramBytes"] == 2 * GIB
    assert result["patches"] == []


@pytest.mark.parametrize("complete", [False, None])
def test_incomplete_owner_cannot_waive_either_cuda_loading_pool(loading_runtime, complete):
    plan, hardware, _, _, _ = loading_runtime
    graph = _graph("group_disk")
    graph["nodes"]["models"]["params"]["weight_loading_mode"] = {"value": "cuda"}
    hardware["systemMemory"]["availableBytes"] = 6 * GIB
    hardware["accelerator"]["freeBytes"] = 3 * GIB
    cache = {"owners": {"models": {
        "cacheKey": workflow_auto_resource.workflow_owner_cache_key(graph, "models"),
        "loadingPhaseReusable": complete, "systemRamBytes": PARAMETERS["totalBytes"],
        "vramBytes": PARAMETERS["totalBytes"],
    }}}
    result = plan(graph, cache_snapshot=cache)
    assert not result["canAutoRun"]
    assert result["reusedOwnerIds"] == []


def test_legacy_graph_missing_loading_control_is_preserved_and_never_secretly_patched(loading_runtime):
    plan, hardware, _, _, _ = loading_runtime
    hardware["systemMemory"]["availableBytes"] = 30 * GIB
    graph = _graph("group_disk")
    before = deepcopy(graph)
    result = plan(graph)
    assert result["canAutoRun"], result["issues"]
    assert result["loaders"][0]["settings"].get("weight_loading_mode", "cpu") == "cpu"
    assert result["patches"] == []
    assert graph == before


@pytest.mark.parametrize('field,value', [('lora_list', []), ('quant_config', {}),
                                       ('unet', {}), ('vae', {}), ('controlnet', {})])
def test_cuda_candidate_matches_runtime_requirement_for_absent_optional_controls(loading_runtime, field, value):
    plan, _, _, _, _ = loading_runtime
    graph = _graph('group_disk')
    graph['nodes']['models']['params']['weight_loading_mode'] = {'value': 'cuda'}
    graph['nodes']['models']['params'][field] = {'value': value}
    before = deepcopy(graph)
    result = plan(graph)
    assert not result['canAutoRun'], 'Auto must not admit CUDA inputs that the actual loader refuses.'
    assert result['patches'] == []
    assert graph == before
