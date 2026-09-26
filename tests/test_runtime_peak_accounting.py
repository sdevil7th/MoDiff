"""Graph high-water marks must survive the existing per-node counter resets."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from modiff.server import WebServer
from modiff.auto_resource import _normalized_measurement


def runtime(kind="cuda"):
    peaks = {0: [100, 120], 1: [200, 240]}
    accelerator = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        reset_peak_memory_stats=lambda index=0: peaks.__setitem__(index, [10, 12]),
        max_memory_allocated=lambda index=0: peaks[index][0],
        max_memory_reserved=lambda index=0: peaks[index][1],
        memory_allocated=lambda index=0: 10,
        memory_reserved=lambda index=0: 12,
    )
    torch = SimpleNamespace(
        cuda=accelerator if kind == "cuda" else SimpleNamespace(is_available=lambda: False),
        xpu=accelerator if kind == "xpu" else None,
        version=SimpleNamespace(hip="test" if kind == "cuda" else None),
    )
    return torch, accelerator, peaks


@pytest.mark.parametrize("kind", ["cuda", "xpu"])
def test_graph_peak_survives_node_reset_and_attempt_reset_isolated(kind):
    server = object.__new__(WebServer)
    torch, accelerator, peaks = runtime(kind)
    with patch("modiff.server.import_module", return_value=torch), patch(
        "modiff.server.reset_memory_stats", side_effect=lambda: accelerator.reset_peak_memory_stats(0)
    ):
        server._reset_runtime_measurement()
        peaks[0] = [300, 400]
        peaks[1] = [500, 600]
        server._reset_node_runtime_measurement()
        assert peaks[0] == [10, 12]
        result = server._runtime_measurement(elapsed_seconds=2)
        assert result["peakAllocatedBytes"] == 300
        assert result["peakReservedBytes"] == 400
        assert result["allocatedBytes"] == 10
        assert result["peakMeasurementVersion"] == 2
        assert server._runtime_memory_peaks[f"{kind}:1"]["peakAllocatedBytes"] == 500
        server._reset_runtime_measurement()
        result = server._runtime_measurement(elapsed_seconds=0)
        assert result["peakAllocatedBytes"] == 10
        assert result["peakReservedBytes"] == 12


def test_last_node_peak_is_not_lost_without_a_following_reset():
    server = object.__new__(WebServer)
    torch, _, peaks = runtime()
    with patch("modiff.server.import_module", return_value=torch):
        server._reset_runtime_measurement()
        peaks[0] = [700, 800]
        result = server._runtime_measurement(elapsed_seconds=1)
    assert result["peakAllocatedBytes"] == 700
    assert result["peakReservedBytes"] == 800


def test_failed_checkpoint_cannot_publish_a_misleading_graph_peak():
    server = object.__new__(WebServer)
    torch, accelerator, _ = runtime()
    with patch("modiff.server.import_module", return_value=torch):
        server._reset_runtime_measurement()
        with patch.object(accelerator, "max_memory_allocated", side_effect=RuntimeError("allocator unavailable")), patch(
            "modiff.server.reset_memory_stats"
        ):
            server._reset_node_runtime_measurement()
        result = server._runtime_measurement(elapsed_seconds=1)
    assert "peakAllocatedBytes" not in result
    assert "peakReservedBytes" not in result
    assert "allocator unavailable" in result["acceleratorMeasurementError"]


def test_non_graph_node_reset_does_not_create_a_phantom_attempt():
    server = object.__new__(WebServer)
    with patch("modiff.server.import_module", side_effect=AssertionError("no active measurement")), patch(
        "modiff.server.reset_memory_stats"
    ) as reset:
        server._reset_node_runtime_measurement()
    reset.assert_called_once_with()
    assert not hasattr(server, "_runtime_memory_peaks")


def test_resource_history_preserves_the_accounting_version():
    assert _normalized_measurement({"peakMeasurementVersion": 2, "peakAllocatedBytes": 300}) == {
        "peakMeasurementVersion": 2, "peakAllocatedBytes": 300,
    }
