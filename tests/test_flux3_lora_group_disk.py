"""Official disk hook + selected LoRA mechanics, tiny CPU fixtures on D scratch."""
from copy import deepcopy
from pathlib import Path
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
def cpu_disk_mechanics(monkeypatch):
    import modiff.diffusers_offload as offload
    from modules.ModularDiffusers import loaders
    original = offload.normalize_offload_mode

    def cpu_mode(mode, auto_offload=True, device=None):
        requested = original(mode, auto_offload=auto_offload)
        if device is not None and torch.device(device).type == "cpu" and requested in {"group_cpu", "group_disk"}:
            return requested
        return original(mode, auto_offload=auto_offload, device=device)

    monkeypatch.setattr(offload, "normalize_offload_mode", cpu_mode)
    monkeypatch.setattr(loaders, "normalize_offload_mode", cpu_mode)


def disk_directories(component):
    leases = vars(component).get("_modiff_disk_offload_leases", ())
    assert leases
    return tuple(str(lease.directory.path) for lease in leases)


@pytest.mark.parametrize("pair", native_tests.PAIRS, ids=lambda row: row[0])
def test_selected_dense_disk_lora_has_fresh_files_and_warm_owner(native_runtime, cpu_disk_mechanics, tmp_path, pair):
    from modiff.config import CONFIG
    graph = placement_graph(pair, mode="group_disk")
    old = native_runtime.run(graph)
    before = native_tests.prediction(native_runtime)
    old_paths = disk_directories(old.loader.transformer)
    former = weakref.ref(old.loader)
    del old
    graph["nodes"]["models"]["params"]["lora_list"] = {
        "value": [native_tests.descriptor(tmp_path, original=True)]}
    node = native_runtime.run(graph)
    assert former() is None
    paths = disk_directories(node.loader.transformer)
    assert not set(paths) & set(old_paths)
    for path in paths:
        Path(path).relative_to(Path(CONFIG.paths["temp"]))
        assert list(Path(path).glob("*.safetensors"))
    changed = native_tests.prediction(native_runtime)
    assert torch.isfinite(changed).all() and not torch.allclose(changed, before)
    owner = weakref.ref(node.loader)
    assert native_runtime.run(deepcopy(graph)).loader is owner()
    assert disk_directories(owner().transformer) == paths
    torch.testing.assert_close(native_tests.prediction(native_runtime), changed, rtol=0, atol=0)
    assert node.loader.vae.dtype == torch.float32


def disk_packed_state(component):
    """Inspect genuine official hook files, not their empty resident placeholders."""
    from diffusers.hooks.group_offloading import GroupOffloadingHook
    from peft.tuners.tuners_utils import BaseTunerLayer
    from safetensors.torch import load_file
    layer = component.get_submodule(native_tests.TARGET)
    if isinstance(layer, BaseTunerLayer):
        layer = layer.get_base_layer()
    groups = {}
    for child in component.modules():
        for hook in getattr(vars(child).get("_diffusers_hook"), "hooks", {}).values():
            if type(hook) is GroupOffloadingHook:
                groups[id(hook.group)] = hook.group
    result = {}
    for name, tensor in [("weight", layer.weight), *layer.named_buffers()]:
        matches = [(group.safetensors_file_path, key)
                   for group in groups.values() for key, value in group.key_to_tensor.items() if value is tensor]
        assert len(matches) == 1
        path, key = matches[0]
        result[name] = load_file(path, device="cpu")[key].reshape(-1).view(torch.uint8).clone()
    return result


@pytest.mark.parametrize("pair", native_tests.PAIRS, ids=lambda row: row[0])
@pytest.mark.parametrize("scheme", ["fp8r", "nvfp4"])
def test_selected_bfl_disk_lora_keeps_exact_packed_file_bytes(bfl_runtime, cpu_disk_mechanics, tmp_path, pair, scheme):
    graph = placement_graph(pair, mode="group_disk", scheme=scheme)
    old = bfl_runtime.run(graph)
    before = native_tests.prediction(bfl_runtime)
    packed = disk_packed_state(old.loader.transformer)
    old_paths = disk_directories(old.loader.transformer)
    former = weakref.ref(old.loader)
    del old
    graph["nodes"]["models"]["params"]["lora_list"] = {
        "value": [native_tests.descriptor(tmp_path, original=True)]}
    node = bfl_runtime.run(graph)
    assert former() is None
    assert not set(disk_directories(node.loader.transformer)) & set(old_paths)
    actual = disk_packed_state(node.loader.transformer)
    assert actual.keys() == packed.keys() and all(torch.equal(actual[key], packed[key]) for key in packed)
    changed = native_tests.prediction(bfl_runtime)
    assert torch.isfinite(changed).all() and not torch.allclose(changed, before)
    owner = weakref.ref(node.loader)
    assert bfl_runtime.run(deepcopy(graph)).loader is owner()
    torch.testing.assert_close(native_tests.prediction(bfl_runtime), changed, rtol=0, atol=0)
    actual = disk_packed_state(node.loader.transformer)
    assert all(torch.equal(actual[key], packed[key]) for key in packed)
    assert node.loader.vae.dtype == torch.float32


def test_disk_scale_remove_and_new_adapter_never_reuse_old_files(native_runtime, cpu_disk_mechanics, tmp_path):
    graph = placement_graph(native_tests.PAIRS[1], mode="group_disk")
    native_runtime.run(graph)
    baseline = native_tests.prediction(native_runtime)
    item = native_tests.descriptor(tmp_path, original=True)
    graph["nodes"]["models"]["params"]["lora_list"] = {"value": [item]}
    native_runtime.run(graph)
    first = native_tests.prediction(native_runtime)
    first_paths = disk_directories(native_runtime.server.node_cache["models"].loader.transformer)
    zero = deepcopy(item)
    zero["scale"] = 0.0
    from modiff.auxiliary_lora import build_lora_descriptor
    zero = build_lora_descriptor(selection={"source": "local", "value": item["artifact"]["root"]},
        weight_name=item["artifact"]["weight_name"], adapter_name="one", scale=0.0,
        revision=None, expected_sha256=None, scheduler_class="", scheduler_config=None)
    graph["nodes"]["models"]["params"]["lora_list"] = {"value": [zero]}
    native_runtime.run(graph)
    torch.testing.assert_close(native_tests.prediction(native_runtime), baseline, rtol=0, atol=0)
    zero_paths = disk_directories(native_runtime.server.node_cache["models"].loader.transformer)
    assert not set(first_paths) & set(zero_paths)
    graph["nodes"]["models"]["params"]["lora_list"] = {"value": []}
    native_runtime.run(graph)
    torch.testing.assert_close(native_tests.prediction(native_runtime), baseline, rtol=0, atol=0)
    second = native_tests.descriptor(tmp_path, name="two", original=True, seed=99)
    graph["nodes"]["models"]["params"]["lora_list"] = {"value": [second]}
    native_runtime.run(graph)
    final = native_tests.prediction(native_runtime)
    assert not torch.allclose(final, first) and not torch.allclose(final, baseline)
    assert not set(disk_directories(native_runtime.server.node_cache["models"].loader.transformer)) & set(first_paths + zero_paths)


def test_changed_owned_disk_file_refuses_warm_binding(native_runtime, cpu_disk_mechanics, tmp_path):
    from modiff.native_disk_lora import disk_lora_binding_matches
    node = native_runtime.run(placement_graph(native_tests.PAIRS[1], [native_tests.descriptor(tmp_path)], mode="group_disk"))
    state = node._native_disk_lora_state
    assert disk_lora_binding_matches(node, state["policy"], native_runtime.manager)
    path = next(Path(disk_directories(node.loader.transformer)[0]).glob("*.safetensors"))
    original = path.read_bytes()
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(bytes([original[-1] ^ 1]))
        handle.flush()
    assert not disk_lora_binding_matches(node, state["policy"], native_runtime.manager)
    assert native_runtime.server.node_cache["models"] is node
    path.write_bytes(original)


@pytest.mark.parametrize("tamper", ["file_path", "tensor_map"])
def test_changed_official_disk_group_mapping_refuses_warm_binding(native_runtime, cpu_disk_mechanics, tmp_path, tamper):
    from modiff.native_disk_lora import disk_lora_binding_matches
    from diffusers.hooks.group_offloading import GroupOffloadingHook
    node = native_runtime.run(placement_graph(native_tests.PAIRS[1], [native_tests.descriptor(tmp_path)], mode="group_disk"))
    state = node._native_disk_lora_state
    groups = {id(hook.group): hook.group for child in node.loader.transformer.modules()
              for hook in getattr(vars(child).get("_diffusers_hook"), "hooks", {}).values()
              if type(hook) is GroupOffloadingHook}
    selected = next(group for group in groups.values() if len(group.key_to_tensor) > 1)
    path = selected.safetensors_file_path
    mapping = selected.key_to_tensor.copy()
    assert disk_lora_binding_matches(node, state["policy"], native_runtime.manager)
    if tamper == "file_path":
        selected.safetensors_file_path = next(group.safetensors_file_path for group in groups.values() if group is not selected)
    else:
        first, second = list(selected.key_to_tensor)[:2]
        selected.key_to_tensor[first], selected.key_to_tensor[second] = selected.key_to_tensor[second], selected.key_to_tensor[first]
    assert not disk_lora_binding_matches(node, state["policy"], native_runtime.manager)
    assert native_runtime.server.node_cache["models"] is node
    selected.safetensors_file_path = path
    selected.key_to_tensor = mapping


@pytest.mark.parametrize("invalid", ["relative-temp", "", None])
def test_invalid_selected_disk_root_never_falls_back_to_working_directory(native_runtime, cpu_disk_mechanics, tmp_path, monkeypatch, invalid):
    from modiff.config import CONFIG
    monkeypatch.setitem(CONFIG.paths, "temp", invalid)
    with pytest.raises((ValueError, RuntimeError), match="storage|absolute"):
        native_runtime.run(placement_graph(native_tests.PAIRS[1], [native_tests.descriptor(tmp_path)], mode="group_disk"))
    assert native_runtime.events == []
