"""Selected LoRA with genuine official hooks, using tiny CPU owners.

The CPU mechanics fixture retains CPU onload/offload and overrides only the
application's accelerator-only eligibility gate so official group hooks run.
It is not CUDA transfer, disk residency, trained output or memory-fit proof.
"""
from copy import deepcopy
import weakref

import pytest
import torch

import test_flux3_native_lora as native_tests
import test_flux3_bfl_lora as bfl_tests
import test_modular_weight_formats as weight_format_tests

native_runtime = native_tests.runtime
artifacts = weight_format_tests.runtime
bfl_runtime = bfl_tests.runtime


def placement_graph(pair, items=None, *, mode="group_cpu", task="text_to_image", scheme="native"):
    graph = native_tests.source_graph(pair, items, task=task)
    params = graph["nodes"]["models"]["params"]
    params["offload_mode"] = {"value": mode}
    params["auto_offload"] = {"value": True}
    params["weight_format"] = {"value": scheme}
    # Current Base spec uses the existing closed guider role.
    if "guider" in graph["nodes"]:
        graph["nodes"]["guider"]["params"].update(
            model_type={"value": pair[0]}, guider={"value": "classifierFreeGuidance"},
            guidance_scale={"value": 3.5},
        )
    return graph


@pytest.fixture
def cpu_group_mechanics(monkeypatch):
    import modiff.diffusers_offload as offload
    from modules.ModularDiffusers import loaders

    original = offload.normalize_offload_mode

    def cpu_mechanics(mode, auto_offload=True, device=None):
        requested = original(mode, auto_offload=auto_offload)
        if device is not None and torch.device(device).type == "cpu" and requested == "group_cpu":
            return requested
        return original(mode, auto_offload=auto_offload, device=device)

    monkeypatch.setattr(offload, "normalize_offload_mode", cpu_mechanics)
    monkeypatch.setattr(loaders, "normalize_offload_mode", cpu_mechanics)


def group_hook_ids(component):
    from diffusers.hooks.group_offloading import GroupOffloadingHook
    hooks = []
    for name, child in component.named_modules():
        registry = vars(child).get("_diffusers_hook")
        for key, hook in getattr(registry, "hooks", {}).items():
            if type(hook) is GroupOffloadingHook:
                assert hook.config.onload_device.type == hook.config.offload_device.type == "cpu"
                hooks.append((name, key, id(hook), id(hook.group)))
    assert hooks, "The actual official group hooks must exist."
    return tuple(hooks)


@pytest.mark.parametrize("pair", native_tests.PAIRS, ids=lambda row: row[0])
@pytest.mark.parametrize("task", ["text_to_image", "edit_image"])
def test_selected_dense_lora_group_cpu_uses_fresh_owned_hooks(native_runtime, cpu_group_mechanics, tmp_path, pair, task):
    graph = placement_graph(pair, task=task)
    old_node = native_runtime.run(graph)
    group_hook_ids(old_node.loader.transformer)
    baseline = native_tests.prediction(native_runtime)
    former = weakref.ref(old_node.loader.transformer)
    del old_node
    item = native_tests.descriptor(tmp_path, original=True)
    requested = placement_graph(pair, [item], task=task)
    node = native_runtime.run(requested)
    assert former() is None
    hooks = group_hook_ids(node.loader.transformer)
    changed = native_tests.prediction(native_runtime)
    assert torch.isfinite(changed).all() and not torch.allclose(changed, baseline)
    owner = weakref.ref(node.loader)
    assert native_runtime.run(deepcopy(requested)).loader is owner()
    assert group_hook_ids(owner().transformer) == hooks
    assert node._native_disk_lora_state["componentComplete"] is True


@pytest.mark.parametrize("pair", native_tests.PAIRS, ids=lambda row: row[0])
@pytest.mark.parametrize("scheme", ["fp8r", "nvfp4"])
def test_selected_bfl_group_cpu_preserves_packed_state_and_warm_owner(bfl_runtime, cpu_group_mechanics, tmp_path, pair, scheme):
    baseline_node = bfl_runtime.run(placement_graph(pair, scheme=scheme))
    group_hook_ids(baseline_node.loader.transformer)
    baseline = native_tests.prediction(bfl_runtime)
    packed = bfl_tests.packed_state(baseline_node.loader.transformer)
    former = weakref.ref(baseline_node.loader)
    del baseline_node
    requested = placement_graph(pair, [native_tests.descriptor(tmp_path, original=True)], scheme=scheme)
    node = bfl_runtime.run(requested)
    assert former() is None
    group_hook_ids(node.loader.transformer)
    changed = native_tests.prediction(bfl_runtime)
    assert torch.isfinite(changed).all() and not torch.allclose(changed, baseline)
    bfl_tests.assert_packed_state(node.loader.transformer, packed)
    owner = weakref.ref(node.loader)
    assert bfl_runtime.run(deepcopy(requested)).loader is owner()
    bfl_tests.assert_packed_state(node.loader.transformer, packed)
    assert node.loader.vae.dtype == torch.float32


@pytest.mark.parametrize("field,value", [
    ("offload_mode", "sequential_cpu"), ("offload_mode", "auto_cpu"),
    ("auto_offload", False), ("auto_offload", 1), ("weight_loading_mode", "cuda"),
    ("unet", "foreign"),
])
def test_unreviewed_or_changed_source_controls_refuse_before_owner_cleanup(native_runtime, cpu_group_mechanics, tmp_path, field, value):
    pair = native_tests.PAIRS[1]
    requested = placement_graph(pair, [native_tests.descriptor(tmp_path)])
    node = native_runtime.run(requested)
    owner = node.loader
    events = len(native_runtime.events)
    changed = deepcopy(requested)
    changed["nodes"]["models"]["params"][field] = {"value": value}
    with pytest.raises((ValueError, RuntimeError)):
        native_runtime.run(changed)
    assert native_runtime.server.node_cache["models"].loader is owner
    assert len(native_runtime.events) == events
    assert node._cache_valid is True


@pytest.mark.parametrize("tamper", ["hook_settings", "typed_context", "adapter_tensor"])
def test_completed_binding_refuses_tampering_without_mutating_owner(native_runtime, cpu_group_mechanics, tmp_path, tamper):
    from modiff.native_disk_lora import disk_lora_binding_matches
    from diffusers.hooks.group_offloading import GroupOffloadingHook
    node = native_runtime.run(placement_graph(native_tests.PAIRS[1], [native_tests.descriptor(tmp_path)]))
    state = node._native_disk_lora_state
    assert disk_lora_binding_matches(node, state["policy"], native_runtime.manager)
    if tamper == "typed_context":
        state["componentPlacementContext"] = object()
    elif tamper == "adapter_tensor":
        with torch.no_grad():
            next(module.lora_A["one"].weight for module in node.loader.transformer.modules()
                 if hasattr(module, "lora_A")).add_(.01)
    else:
        for child in node.loader.transformer.modules():
            for hook in getattr(vars(child).get("_diffusers_hook"), "hooks", {}).values():
                if type(hook) is GroupOffloadingHook:
                    hook.config.non_blocking = True
    assert not disk_lora_binding_matches(node, state["policy"], native_runtime.manager)
    assert native_runtime.server.node_cache["models"] is node


def test_base_source_guider_cannot_borrow_another_model(native_runtime, cpu_group_mechanics, tmp_path):
    source = placement_graph(native_tests.PAIRS[0], [native_tests.descriptor(tmp_path)])
    source["nodes"]["guider"]["params"]["model_type"] = {"value": native_tests.PAIRS[1][0]}
    with pytest.raises(ValueError, match="guidance"):
        native_runtime.run(source)
    assert native_runtime.events == []
