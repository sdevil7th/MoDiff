"""Host staging must fit before selecting official asynchronous group hooks."""
from unittest.mock import patch

import pytest
import torch

from modiff.diffusers_offload import (
    MIN_STREAMING_RAM_BYTES,
    _apply_group_to_module,
    _streaming_component_bytes,
    _streaming_offload_allowed,
)


def _hook_arguments(module, available, *, mode="group_cpu", allow_streaming=True, deterministic=False):
    calls = []
    with patch("modiff.hardware.system_memory_snapshot", return_value={"available_bytes": available}), \
            patch("torch.are_deterministic_algorithms_enabled", return_value=deterministic), \
            patch("diffusers.hooks.apply_group_offloading", side_effect=lambda _module, **kwargs: calls.append(kwargs)):
        result = _apply_group_to_module(module, component_name="text_encoder", device="cuda:0",
            node_id="staging-regression", scope="test", mode=mode, allow_streaming=allow_streaming)
    assert result == "text_encoder" and len(calls) == 1
    return calls[0]


def test_streaming_keeps_available_ram_floor_after_host_staging():
    # Real small CPU weights reproduce the selection bug without allocating a
    # multi-gigabyte model. Upstream retains CPU source tensors and creates
    # pinned transfer material; reserve two full component copies conservatively.
    module = torch.nn.Linear(64, 64, bias=False)
    weight_bytes = module.weight.untyped_storage().nbytes()
    arguments = _hook_arguments(module, MIN_STREAMING_RAM_BYTES + 2 * weight_bytes - 1)
    assert arguments["use_stream"] is False
    assert arguments["non_blocking"] is False
    assert arguments["low_cpu_mem_usage"] is True


def test_small_component_can_stream_when_host_staging_and_floor_fit():
    module = torch.nn.Linear(64, 64, bias=False)
    weight_bytes = module.weight.untyped_storage().nbytes()
    arguments = _hook_arguments(module, MIN_STREAMING_RAM_BYTES + 2 * weight_bytes)
    assert arguments["use_stream"] is True
    assert arguments["non_blocking"] is True


@pytest.mark.parametrize("available,allow_streaming,deterministic", [
    (MIN_STREAMING_RAM_BYTES - 1, True, False),
    (32 * 1024**3, False, False),
    (32 * 1024**3, True, True),
    (True, True, False),
    (None, True, False),
])
def test_unavailable_or_disabled_streaming_keeps_synchronous_hooks(available, allow_streaming, deterministic):
    arguments = _hook_arguments(torch.nn.Linear(2, 2), available,
        allow_streaming=allow_streaming, deterministic=deterministic)
    assert arguments["use_stream"] is False
    assert arguments["non_blocking"] is False


def test_full_underlying_storage_is_reserved_for_a_small_registered_view():
    backing = torch.zeros(1024)
    module = torch.nn.Module()
    module.register_buffer("view", backing[:1])
    assert _streaming_component_bytes(module) == backing.untyped_storage().nbytes()


def test_independently_pinned_views_keep_logical_coverage_after_storage_deduplication():
    backing = torch.zeros(1024)
    module = torch.nn.Module()
    for name in ("one", "two", "three"):
        module.register_buffer(name, backing[:])
    assert _streaming_component_bytes(module) == 3 * backing.untyped_storage().nbytes()


def test_native_registered_state_counts_actual_dtype_buffers_and_avoids_custom_traversal():
    class TraversalSentinel(torch.nn.Linear):
        def parameters(self, *args, **kwargs):
            raise AssertionError("Do not invoke model traversal overrides for staging inspection.")

        buffers = named_parameters = named_buffers = modules = named_modules = parameters

    module = TraversalSentinel(4, 4, bias=False, dtype=torch.bfloat16)
    module.register_buffer("packed", torch.zeros(7, dtype=torch.uint8))
    assert _streaming_component_bytes(module) == 4 * 4 * 2 + 7


@pytest.mark.parametrize("kind", ["meta", "subclass", "non_native_mapping"])
def test_unknown_or_unmaterialized_storage_refuses_streaming(kind):
    module = torch.nn.Module()
    if kind == "meta":
        module.register_buffer("unknown", torch.empty(3, device="meta"))
    elif kind == "subclass":
        class TensorSubclass(torch.Tensor):
            pass
        module.register_buffer("unknown", torch.zeros(3).as_subclass(TensorSubclass))
    else:
        module._buffers = []
    assert _streaming_component_bytes(module) is None
    assert _hook_arguments(module, 64 * 1024**3)["use_stream"] is False


def test_module_and_tensor_inventory_limits_are_checked_incrementally():
    module = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    with patch("modiff.diffusers_offload._MAX_STREAMING_MODULES", 1):
        assert _streaming_component_bytes(module) is None
    with patch("modiff.diffusers_offload._MAX_STREAMING_TENSOR_REFERENCES", 1):
        assert _streaming_component_bytes(module) is None


@pytest.mark.parametrize("device,mode", [("cpu", "group_cpu"), ("cuda:0", "group_disk")])
def test_other_placements_do_not_inspect_staging_or_hardware(device, mode):
    with patch("modiff.diffusers_offload._streaming_component_bytes", side_effect=AssertionError("no census")), \
            patch("modiff.hardware.system_memory_snapshot", side_effect=AssertionError("no hardware query")):
        assert _streaming_offload_allowed(device, mode, torch.nn.Linear(2, 2)) is False
