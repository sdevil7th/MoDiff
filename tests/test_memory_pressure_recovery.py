"""Fault-injected pressure and ownership tests without exhausting host RAM."""
from unittest.mock import patch

import pytest
import torch

from modiff.NodeBase import NodeBase
from utils.memory_menager import GIB, MemoryManager


class Model:
    def __init__(self, device="cpu"):
        self.device = device
        self.moves = []

    def to(self, device):
        self.moves.append(device)
        self.device = str(device)
        return self


@pytest.fixture
def manager():
    with patch("utils.memory_menager.memory_flush"), patch(
        "modiff.hardware.system_memory_snapshot", return_value={"available_bytes": 32 * GIB, "total_bytes": 64 * GIB},
    ):
        instance = MemoryManager()
        instance.policy = "lru"
        yield instance
        instance.clear()


def test_discard_does_not_materialize_a_cpu_copy(manager):
    model = Model("cuda:0")
    identity = manager.add(model)
    assert manager.remove(identity) == identity
    assert model.moves == [] and not manager.cache


def test_exec_excludes_entire_working_set_before_any_component_load(manager):
    first, second, inactive = (manager.add(Model()) for _ in range(3))
    calls = []
    def load(identity, device, exclude):
        assert {first, second} <= set(exclude)
        assert {first, second} <= set(manager._active_ids)
        calls.append(identity)
    with patch.object(manager, "load_model", side_effect=load):
        assert manager.exec(lambda: "done", "cuda:0", models=[first, second]) == "done"
    assert calls == [first, second] and not manager._active_ids
    assert inactive in manager.cache


def test_pressure_inside_dispatch_cannot_remove_active_cpu_components(manager):
    active = manager.add(Model())
    inactive_model = Model()
    inactive = manager.add(inactive_model)
    readings = [{"available_bytes": GIB, "total_bytes": 64 * GIB},
                {"available_bytes": 32 * GIB, "total_bytes": 64 * GIB}]
    def execute():
        with patch("modiff.hardware.system_memory_snapshot", side_effect=readings):
            added = manager.add(Model())
        assert active in manager.cache and added in manager.cache and inactive not in manager.cache
        assert inactive_model.moves == []
        return added
    manager.exec(execute, "cpu", models=[active])
    assert not manager._active_ids


def test_proactive_accelerator_pressure_evicts_inactive_not_prior_active_model(manager):
    first, second, inactive = Model("cuda:0"), Model(), Model("cuda:0")
    first_id, second_id = manager.add(first), manager.add(second)
    manager.add(inactive, priority=2)
    manager.cache[second_id]["size"] = 8 * GIB
    with patch("utils.memory_menager.torch.cuda.mem_get_info", side_effect=[(0, 16 * GIB), (10 * GIB, 16 * GIB)]):
        result = manager.exec(lambda: (first.device, second.device), "cuda:0", models=[first_id, second_id])
    assert result == ("cuda:0", "cuda:0")
    assert first.moves == [] and inactive.moves == ["cpu"]


def test_failed_load_releases_leases_and_cleanup_does_not_hide_original_error(manager):
    identity = manager.add(Model())
    manager.policy = "no_cache"
    failure = torch.OutOfMemoryError("original allocation failure")
    with patch.object(manager, "load_model", side_effect=failure), patch.object(
        manager, "unload_model", side_effect=RuntimeError("device lost during cleanup"),
    ):
        with pytest.raises(torch.OutOfMemoryError) as error:
            manager.exec(lambda: pytest.fail("must not invoke"), "cuda:0", models=[identity])
    assert error.value is failure and not manager._active_ids


def test_nested_no_cache_dispatch_does_not_offload_outer_active_owner(manager):
    identity = manager.add(Model())
    manager.policy = "no_cache"
    with patch.object(manager, "unload_model") as unload:
        def outer():
            manager.exec(lambda: None, "cpu", models=[identity])
            unload.assert_not_called()
        manager.exec(outer, "cpu", models=[identity])
    unload.assert_called_once_with(identity)
    assert not manager._active_ids


def test_oom_callback_is_not_replayed_with_mutated_generator(manager):
    manager.add(Model("cpu"))
    generator = torch.Generator().manual_seed(9)
    seen = []
    def execute():
        seen.append(torch.rand(1, generator=generator))
        raise torch.OutOfMemoryError("activation allocation failed")
    with pytest.raises(torch.OutOfMemoryError):
        manager.exec(execute, "cpu")
    assert len(seen) == 1 and not manager._active_ids


def test_sensor_failure_after_pressure_eviction_is_bounded_and_nonfatal(manager):
    first = manager.add(Model())
    second = manager.add(Model())
    with patch("modiff.hardware.system_memory_snapshot", side_effect=[
        {"available_bytes": GIB, "total_bytes": 64 * GIB}, OSError("sensor unavailable"),
    ]) as sensor:
        assert manager._evict_system_ram_pressure() == [first]
    assert sensor.call_count == 2 and second in manager.cache


@pytest.mark.parametrize("available", [None, True, -1, "0"])
def test_invalid_pressure_reading_cannot_evict_models(manager, available):
    identity = manager.add(Model())
    with patch("modiff.hardware.system_memory_snapshot", return_value={"available_bytes": available}):
        assert manager._evict_system_ram_pressure() == []
    assert identity in manager.cache


def test_node_reloads_evicted_handle_once_instead_of_returning_a_stale_cache_hit(manager):
    class Loader(NodeBase):
        def execute(self, **kwargs):
            self.calls += 1
            return {"model": self.mm_add(Model())}
    module = ".".join(Loader.__module__.split(".")[:-1])
    definitions = {module: {"Loader": {"skipParamsCheck": True, "params": {}}}}
    with patch("modiff.NodeBase._module_map", return_value=definitions), patch("modiff.NodeBase.memory_manager", manager):
        loader = Loader("pressure-loader")
        loader.calls = 0
        first = loader()["model"]
        assert loader()["model"] == first and loader.calls == 1
        manager.remove(first)
        second = loader()["model"]
        assert second != first and loader.calls == 2 and loader._cache_reason == "models_evicted"
        assert loader()["model"] == second and loader.calls == 2
        loader._mm_models = []
