from types import SimpleNamespace
from unittest.mock import patch

import pytest

from modiff import auto_resource as auto
from modiff.hardware import get_hardware_snapshot
from modiff.workflow_auto_lifecycle import assert_next_owner_capacity

GIB = 1024 ** 3

def normalized(device):
    return {
        "devices": [device], "torch": {},
        "system": {"ram_total": 64 * GIB, "ram_available": 48 * GIB},
        "disk": {"path": "private-test", "free_bytes": 128 * GIB},
    }


def device(kind, *, planning=None, legacy=None, process=12 * GIB, memory_kind="dedicated"):
    return {
        "type": kind, "name": "Mock device", "memory_kind": memory_kind,
        "planning_memory_total": 16 * GIB, "vram_total": 16 * GIB,
        "torch_vram_total": 16 * GIB, "planning_memory_free": planning,
        "vram_free": legacy, "torch_vram_free": process,
    }


@pytest.mark.parametrize("kind", ["cuda", "xpu", "mps"])
@pytest.mark.parametrize("planning,legacy", [(0, 9 * GIB), (0, None), ("0", 9 * GIB)])
def test_known_zero_planning_memory_has_priority(kind, planning, legacy):
    accelerator = auto._normalized_accelerator_snapshot(normalized(device(kind, planning=planning, legacy=legacy)))
    assert accelerator["freeBytes"] == 0


@pytest.mark.parametrize("kind", ["cuda", "xpu"])
def test_legacy_known_zero_is_not_replaced_by_process_capacity(kind):
    accelerator = auto._normalized_accelerator_snapshot(normalized(device(kind, planning=None, legacy=0)))
    assert accelerator["freeBytes"] == 0


@pytest.mark.parametrize("kind", ["cuda", "xpu", "mps"])
def test_unknown_planning_memory_retains_legacy_fallback(kind):
    accelerator = auto._normalized_accelerator_snapshot(normalized(device(kind, planning=None, legacy=5 * GIB)))
    assert accelerator["freeBytes"] == 5 * GIB


@pytest.mark.parametrize("kind", ["cuda", "xpu"])
def test_unknown_planning_and_legacy_memory_retains_process_fallback(kind):
    accelerator = auto._normalized_accelerator_snapshot(normalized(device(kind)))
    assert accelerator["freeBytes"] == 12 * GIB


def test_mps_unknown_memory_does_not_add_a_new_process_fallback():
    accelerator = auto._normalized_accelerator_snapshot(normalized(device("mps")))
    assert accelerator["freeBytes"] is None


class OccupiedXpu:
    def is_available(self): return True
    def device_count(self): return 1
    def get_device_name(self, _index): return "Mock Intel Arc"
    def get_device_properties(self, _index): return SimpleNamespace(total_memory=16 * GIB)
    def mem_get_info(self, _index): return 0, 16 * GIB
    def memory_allocated(self, _index): return 0
    def memory_reserved(self, _index): return 0


def test_actual_xpu_hardware_probe_zero_blocks_the_next_owner(tmp_path):
    torch = SimpleNamespace(
        __version__="mock", version=SimpleNamespace(cuda=None, hip=None),
        cuda=SimpleNamespace(is_available=lambda: False), xpu=OccupiedXpu(),
        backends=SimpleNamespace(mps=SimpleNamespace(is_built=lambda: False, is_available=lambda: False)),
    )
    with patch("modiff.hardware.system_memory_snapshot", return_value={"total_bytes": 64 * GIB, "available_bytes": 48 * GIB}):
        hardware = get_hardware_snapshot(torch_module=torch)
    probe = hardware["devices"][0]
    assert probe["planning_memory_free"] == probe["vram_free"] == 0
    assert probe["torch_vram_free"] == 16 * GIB  # Process-only capacity cannot override the driver.
    snapshot = auto._hardware_snapshot({"hardware": hardware}, tmp_path)
    owner = {"ownerId": "next-image-model", "device": "xpu:0", "requirements": {"systemRamBytes": GIB, "vramBytes": GIB}}
    with pytest.raises(ValueError, match="actual free memory"):
        assert_next_owner_capacity(None, owner, snapshot)


@pytest.mark.parametrize("shared_free,blocked", [(0, True), (6 * GIB, False)])
def test_shared_pool_capacity_rules_are_preserved(tmp_path, shared_free, blocked):
    shared_device = device("cuda", planning=0, legacy=0, memory_kind="shared")
    shared_device["shared_memory_free"] = shared_free
    snapshot = auto._hardware_snapshot({"hardware": normalized(shared_device)}, tmp_path)
    owner = {"ownerId": "shared-image-model", "device": "cuda:0", "requirements": {"systemRamBytes": GIB, "vramBytes": GIB}}
    if blocked:
        with pytest.raises(ValueError, match="actual free memory"):
            assert_next_owner_capacity(None, owner, snapshot)
    else:
        assert_next_owner_capacity(None, owner, snapshot)
