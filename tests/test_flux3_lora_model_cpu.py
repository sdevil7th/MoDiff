"""Genuine manager CPU-offload mechanics; CPU onload is not CUDA/fit proof."""
from copy import deepcopy
import weakref

import pytest
import torch

import test_flux3_native_lora as native_tests
import test_flux3_bfl_lora as bfl_tests
import test_modular_weight_formats as weight_format_tests
from test_flux3_lora_placement import placement_graph

native_runtime = native_tests.runtime
artifacts = weight_format_tests.runtime
bfl_runtime = bfl_tests.runtime


@pytest.fixture
def cpu_manager_mechanics(monkeypatch):
    import psutil
    import modiff.diffusers_offload as offload
    from modules.ModularDiffusers import loaders
    from diffusers.utils.torch_utils import TorchDeviceBackend

    original_mode = offload.normalize_offload_mode
    original_memory = TorchDeviceBackend.mem_get_info

    def cpu_mode(mode, auto_offload=True, device=None):
        requested = original_mode(mode, auto_offload=auto_offload)
        if device is not None and torch.device(device).type == "cpu" and requested == "model_cpu":
            return requested
        return original_mode(mode, auto_offload=auto_offload, device=device)

    def cpu_memory(backend):
        if backend.device.type == "cpu":
            memory = psutil.virtual_memory()
            return memory.available, memory.total
        return original_memory(backend)

    monkeypatch.setattr(offload, "normalize_offload_mode", cpu_mode)
    monkeypatch.setattr(loaders, "normalize_offload_mode", cpu_mode)
    monkeypatch.setattr(TorchDeviceBackend, "mem_get_info", cpu_memory)


def manager_hook(runtime, component):
    from diffusers.modular_pipelines.components_manager import CustomOffloadHook, UserCustomOffloadHook
    hook = vars(component).get("_hf_hook")
    assert type(hook) is CustomOffloadHook
    selected = [row for row in runtime.manager.model_hooks
                if type(row) is UserCustomOffloadHook and row.model is component and row.hook is hook]
    assert len(selected) == 1
    assert runtime.manager.components[selected[0].model_id] is component
    assert selected[0].model_id in runtime.manager.collections["models"]
    assert hook.execution_device.type == "cpu"
    return id(hook)


@pytest.mark.parametrize("pair", native_tests.PAIRS, ids=lambda row: row[0])
@pytest.mark.parametrize("task", ["text_to_image", "edit_image"])
def test_selected_dense_model_cpu_lora_keeps_genuine_manager_hook(native_runtime, cpu_manager_mechanics, tmp_path, pair, task):
    graph = placement_graph(pair, mode="model_cpu", task=task)
    baseline_node = native_runtime.run(graph)
    manager_hook(native_runtime, baseline_node.loader.transformer)
    baseline = native_tests.prediction(native_runtime)
    former = weakref.ref(baseline_node.loader)
    del baseline_node
    graph["nodes"]["models"]["params"]["lora_list"] = {
        "value": [native_tests.descriptor(tmp_path, original=True)]}
    node = native_runtime.run(graph)
    assert former() is None
    hook = manager_hook(native_runtime, node.loader.transformer)
    result = native_tests.prediction(native_runtime)
    assert torch.isfinite(result).all() and not torch.allclose(result, baseline)
    owner = weakref.ref(node.loader)
    assert native_runtime.run(deepcopy(graph)).loader is owner()
    assert manager_hook(native_runtime, owner().transformer) == hook


@pytest.mark.parametrize("pair", native_tests.PAIRS, ids=lambda row: row[0])
@pytest.mark.parametrize("scheme", ["fp8r", "nvfp4"])
def test_selected_bfl_model_cpu_lora_preserves_packed_state(bfl_runtime, cpu_manager_mechanics, tmp_path, pair, scheme):
    graph = placement_graph(pair, mode="model_cpu", scheme=scheme)
    baseline_node = bfl_runtime.run(graph)
    manager_hook(bfl_runtime, baseline_node.loader.transformer)
    baseline = native_tests.prediction(bfl_runtime)
    packed = bfl_tests.packed_state(baseline_node.loader.transformer)
    former = weakref.ref(baseline_node.loader)
    del baseline_node
    graph["nodes"]["models"]["params"]["lora_list"] = {
        "value": [native_tests.descriptor(tmp_path, original=True)]}
    node = bfl_runtime.run(graph)
    assert former() is None
    hook = manager_hook(bfl_runtime, node.loader.transformer)
    changed = native_tests.prediction(bfl_runtime)
    assert torch.isfinite(changed).all() and not torch.allclose(changed, baseline)
    bfl_tests.assert_packed_state(node.loader.transformer, packed)
    owner = weakref.ref(node.loader)
    assert bfl_runtime.run(deepcopy(graph)).loader is owner()
    assert manager_hook(bfl_runtime, owner().transformer) == hook
    bfl_tests.assert_packed_state(owner().transformer, packed)
    assert owner().vae.dtype == torch.float32


@pytest.mark.parametrize("collection", [None, "unrelated"])
def test_foreign_neural_owner_refuses_before_any_manager_hook_change(native_runtime, cpu_manager_mechanics, tmp_path, monkeypatch, collection):
    foreign = torch.nn.Linear(2, 2)
    before = foreign.weight.detach().clone()
    identities = []
    original = native_runtime.loaders.ModelsLoader.execute

    def after_initial_orphan_cleanup(node, **kwargs):
        # The ordinary graph startup sweeps unattached manager records. Insert
        # the competing owner at the actual execute boundary to exercise the
        # pre-configuration guard, rather than that unrelated startup sweep.
        identities.append(native_runtime.manager.add("foreign", foreign, collection=collection))
        return original(node, **kwargs)

    monkeypatch.setattr(native_runtime.loaders.ModelsLoader, "execute", after_initial_orphan_cleanup)
    with pytest.raises(RuntimeError, match="another neural owner"):
        native_runtime.run(placement_graph(native_tests.PAIRS[1], [native_tests.descriptor(tmp_path)], mode="model_cpu"))
    assert native_runtime.events == []
    assert len(identities) == 1 and native_runtime.manager.components[identities[0]] is foreign
    assert native_runtime.manager._auto_offload_enabled is False
    assert vars(foreign).get("_hf_hook") is None
    torch.testing.assert_close(foreign.weight, before, rtol=0, atol=0)


@pytest.mark.parametrize("tamper", ["wrapper", "strategy", "nested_hook", "peer_list"])
def test_model_cpu_seal_refuses_changed_hook_identity(native_runtime, cpu_manager_mechanics, tmp_path, tamper):
    from modiff.native_disk_lora import disk_lora_binding_matches
    node = native_runtime.run(placement_graph(native_tests.PAIRS[1], [native_tests.descriptor(tmp_path)], mode="model_cpu"))
    state = node._native_disk_lora_state
    assert disk_lora_binding_matches(node, state["policy"], native_runtime.manager)
    hook = vars(node.loader.transformer)["_hf_hook"]
    wrapper = next(row for row in native_runtime.manager.model_hooks if row.model is node.loader.transformer)
    original_id = wrapper.model_id
    original_margin = native_runtime.manager._offload_strategy.memory_reserve_margin
    original_peers = hook.other_hooks
    if tamper == "wrapper":
        wrapper.model_id = "foreign"
    elif tamper == "strategy":
        native_runtime.manager._offload_strategy.memory_reserve_margin += 1
    elif tamper == "nested_hook":
        vars(node.loader.transformer.mode_blocks["image"][0])["_diffusers_hook"] = object()
    else:
        hook.other_hooks = []
    assert not disk_lora_binding_matches(node, state["policy"], native_runtime.manager)
    assert native_runtime.server.node_cache["models"] is node
    # Restore the intentional corruption before ordinary destructor cleanup.
    wrapper.model_id = original_id
    native_runtime.manager._offload_strategy.memory_reserve_margin = original_margin
    hook.other_hooks = original_peers
    if tamper == "nested_hook":
        del vars(node.loader.transformer.mode_blocks["image"][0])["_diffusers_hook"]
