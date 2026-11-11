"""Actual tiny CPU hooks, not model-fit or accelerator qualification."""
import sys
from types import SimpleNamespace

import pytest
import torch
from accelerate import cpu_offload, disk_offload
from accelerate.hooks import AlignDevicesHook, SequentialHook
from accelerate.utils.offload import PrefixedDataset

from modiff.server import WebServer, memory_manager


def cache_owner(model, monkeypatch, *, alias=False):
    class Loader:
        pass

    app = object.__new__(WebServer)
    app.node_cache = {}
    monkeypatch.setattr(memory_manager, "cache", {"weights": {"model": model}})
    monkeypatch.setitem(sys.modules, "modules.ModularDiffusers", SimpleNamespace(
        components=SimpleNamespace(collections={}, components={})))
    for name in ("load", "alias") if alias else ("load",):
        cached = Loader()
        cached._cache_valid, cached._cache_invalidated = True, False
        cached._mm_models, cached.output = ["weights"], {"model": model}
        cached.loader = model
        app.node_cache[name] = cached
        app._record_workflow_auto_owner({"nodeId": name, "cacheKey": name + "-key"})
    return app


def offload_hook(model):
    hook = model._hf_hook
    if type(hook) is SequentialHook:
        hook, = [item for item in hook.hooks if type(item) is AlignDevicesHook and item.offload]
    assert type(hook) is AlignDevicesHook and hook.offload
    return hook


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("offload_buffers", [False, True])
def test_real_cpu_offload_weights_remain_live_storage_after_meta_replacement(
    monkeypatch, nested, offload_buffers,
):
    linear = torch.nn.Linear(4, 4, bias=False)
    # An alias must not double the physical storage, whether held by a Module
    # buffer or by the official offload hook's CPU state dictionary.
    linear.register_buffer("weight_alias", linear.weight.detach())
    model = torch.nn.Sequential(linear) if nested else linear
    cpu_offload(model, execution_device=torch.device("cpu"), offload_buffers=offload_buffers)
    assert linear.weight.device.type == "meta"
    app = cache_owner(model, monkeypatch, alias=True)
    for _ in range(2):
        snapshot = app._workflow_auto_cache_snapshot()
        assert snapshot["reclaimable"] == {"systemRamBytes": 64, "vramBytes": 0}
        assert all(owner["systemRamBytes"] == 64 for owner in snapshot["owners"].values())
        assert all(len(owner["weightStorage"]) == 1 for owner in snapshot["owners"].values())
        assert linear.weight.device.type == "meta", "inspection materialized offloaded weights"
        model(torch.ones(1, 4))


def test_real_cpu_hook_storage_is_not_reuse_credit_for_an_invalidated_owner(monkeypatch):
    model = cpu_offload(torch.nn.Linear(4, 4, bias=False), execution_device=torch.device("cpu"))
    app = cache_owner(model, monkeypatch)
    app.node_cache["load"]._cache_invalidated = True
    snapshot = app._workflow_auto_cache_snapshot()
    assert snapshot["owners"] == {}
    assert snapshot["reclaimable"] == {"systemRamBytes": 64, "vramBytes": 0}


def test_cpu_hook_prefix_does_not_credit_unrelated_dataset_values(monkeypatch):
    model = cpu_offload(torch.nn.Linear(4, 4, bias=False), execution_device=torch.device("cpu"))
    hook = offload_hook(model)
    hook.weights_map.dataset["unrelated"] = torch.ones(100)
    app = cache_owner(model, monkeypatch)
    assert app._workflow_auto_cache_snapshot()["owners"]["load"]["systemRamBytes"] == 64


def test_nested_offload_prefixes_resolve_only_the_actual_cpu_tensor(monkeypatch):
    model = cpu_offload(torch.nn.Linear(4, 4, bias=False), execution_device=torch.device("cpu"))
    hook = offload_hook(model)
    weight = hook.weights_map["weight"]
    hook.weights_map = PrefixedDataset(PrefixedDataset({"outer.inner.weight": weight}, "outer."), "inner.")
    app = cache_owner(model, monkeypatch)
    assert app._workflow_auto_cache_snapshot()["owners"]["load"]["systemRamBytes"] == 64


def test_hook_inspection_never_calls_an_unknown_mapping(monkeypatch):
    class UnknownMapping(dict):
        def get(self, *args):
            pytest.fail("unknown mapping executed")

        def __getitem__(self, key):
            pytest.fail("unknown mapping executed")

        def __iter__(self):
            pytest.fail("unknown mapping executed")

        def values(self):
            pytest.fail("unknown mapping executed")

    model = cpu_offload(torch.nn.Linear(4, 4, bias=False), execution_device=torch.device("cpu"))
    offload_hook(model).weights_map = PrefixedDataset(UnknownMapping(), "")
    app = cache_owner(model, monkeypatch)
    assert app._workflow_auto_cache_snapshot()["owners"]["load"]["systemRamBytes"] == 0


def test_real_disk_hook_does_not_read_or_credit_lazy_weights(monkeypatch, tmp_path):
    model = torch.nn.Linear(4, 4, bias=False)
    disk_offload(model, str(tmp_path), execution_device=torch.device("cpu"))
    hook = offload_hook(model)
    dataset = hook.weights_map.dataset
    monkeypatch.setattr(type(dataset), "__getitem__", lambda *args: pytest.fail("disk weight read"))
    app = cache_owner(model, monkeypatch)
    assert app._workflow_auto_cache_snapshot()["reclaimable"] == {"systemRamBytes": 0, "vramBytes": 0}


def test_a_custom_hook_with_similar_fields_is_not_executed_or_credited(monkeypatch):
    model = torch.nn.Linear(4, 4, bias=False, device="meta")
    model._hf_hook = SimpleNamespace(
        offload=True, original_devices={"weight": torch.device("cpu")},
        weights_map={"weight": torch.ones(4, 4)},
    )
    app = cache_owner(model, monkeypatch)
    assert app._workflow_auto_cache_snapshot()["owners"]["load"]["systemRamBytes"] == 0


def test_unknown_hook_state_descriptor_is_never_inspected(monkeypatch):
    class CustomHook:
        @property
        def __dict__(self):
            pytest.fail("custom hook descriptor executed")

    model = torch.nn.Linear(4, 4, bias=False, device="meta")
    model._hf_hook = CustomHook()
    app = cache_owner(model, monkeypatch)
    assert app._workflow_auto_cache_snapshot()["reclaimable"] == {"systemRamBytes": 0, "vramBytes": 0}


def test_hook_storage_snapshot_does_not_retain_the_model(monkeypatch):
    import gc
    import weakref

    model = cpu_offload(torch.nn.Linear(4, 4, bias=False), execution_device=torch.device("cpu"))
    reference = weakref.ref(model)
    app = cache_owner(model, monkeypatch)
    snapshot = app._workflow_auto_cache_snapshot()
    assert snapshot["owners"]["load"]["systemRamBytes"] == 64
    app.node_cache.clear()
    memory_manager.cache.clear()
    del model
    gc.collect()
    assert reference() is None
