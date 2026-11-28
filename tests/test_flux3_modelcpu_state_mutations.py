"""Actual PEFT CPU owner controls; CUDA transition is proved separately."""
from types import SimpleNamespace

import pytest
import torch

import test_flux3_native_lora as native_tests
import test_modular_weight_formats as weight_format_tests
import test_flux3_bfl_lora as bfl_tests
import test_flux3_lora_model_cpu as model_cpu_tests
from test_flux3_lora_placement import placement_graph

native_runtime = native_tests.runtime
artifacts = weight_format_tests.runtime
bfl_runtime = bfl_tests.runtime
cpu_manager_mechanics = model_cpu_tests.cpu_manager_mechanics


@pytest.mark.parametrize("scheme", ["native", "nvfp4"])
@pytest.mark.parametrize("mutation", ["version", "identity", "dtype", "shape", "scale", "order", "foreign_device"])
def test_model_cpu_device_normalization_preserves_semantic_refusals(request, cpu_manager_mechanics, tmp_path, scheme, mutation):
    from modiff.native_disk_lora import disk_lora_binding_matches
    runtime = request.getfixturevalue("native_runtime" if scheme == "native" else "bfl_runtime")
    graph = placement_graph(native_tests.PAIRS[1],
        [native_tests.descriptor(tmp_path), native_tests.descriptor(tmp_path, name="two", scale=.35)],
        mode="model_cpu", scheme=scheme)
    node = runtime.run(graph)
    policy = node._native_disk_lora_state["policy"]
    sealed = node._native_disk_lora_state["componentAdapterState"]
    assert disk_lora_binding_matches(node, policy, runtime.manager)
    layer = node.loader.transformer.get_submodule(native_tests.TARGET)
    factor = layer.lora_A["one"]
    original = factor.weight
    original_data = original.data
    original_scale = layer.scaling["one"]
    original_order = list(layer.active_adapters)
    try:
        if mutation == "version":
            with torch.no_grad():
                factor.weight.add_(.1)
        elif mutation == "identity":
            factor.weight = torch.nn.Parameter(factor.weight.detach().clone())
        elif mutation == "dtype":
            factor.weight.data = factor.weight.data.to(torch.float16)
        elif mutation == "shape":
            factor.weight.data = factor.weight.data[:, :-1].clone()
        elif mutation == "scale":
            layer.scaling["one"] = original_scale * .5
        elif mutation == "order":
            layer.set_adapter(["two", "one"])
        else:
            factor.weight = torch.nn.Parameter(factor.weight.detach().to("meta"))
        assert not disk_lora_binding_matches(node, policy, runtime.manager)
        assert node._native_disk_lora_state["componentAdapterState"] is sealed
        assert runtime.server.node_cache["models"] is node
    finally:
        factor.weight = original
        factor.weight.data = original_data
        layer.scaling["one"] = original_scale
        layer.set_adapter(original_order)


def test_untyped_context_cannot_acquire_placement_normalization(native_runtime, tmp_path):
    from modiff.native_component_lora import _adapter_execution_state
    node = native_runtime.run(native_tests.source_graph(native_tests.PAIRS[1], [native_tests.descriptor(tmp_path)]))
    component = node.loader.transformer
    strict = _adapter_execution_state(component)
    fake = SimpleNamespace(_selected_component_lora_context=SimpleNamespace(
        placement=SimpleNamespace(mode="model_cpu", device="cuda:0")))
    assert _adapter_execution_state(component, node=fake) == strict


def test_none_placement_keeps_existing_strict_adapter_snapshot(native_runtime, tmp_path):
    from modiff.native_component_lora import _adapter_execution_state
    # This plain NONE owner must retain the exact existing strict snapshot.
    node = native_runtime.run(native_tests.source_graph(native_tests.PAIRS[1], [native_tests.descriptor(tmp_path)]))
    assert _adapter_execution_state(node.loader.transformer, node=node) == _adapter_execution_state(node.loader.transformer)
