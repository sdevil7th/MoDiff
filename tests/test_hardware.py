import json
import sys
import types
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modiff.hardware import SCHEMA_VERSION, get_hardware_snapshot  # noqa: E402


GIB = 1024**3


def _raise(message):
    raise RuntimeError(message)


def _fake_torch(*, cuda, mps_backend=None, mps_runtime=None):
    return types.SimpleNamespace(
        __version__="test-torch",
        cuda=cuda,
        backends=types.SimpleNamespace(mps=mps_backend),
        mps=mps_runtime,
    )


class HardwareSnapshotTests(unittest.TestCase):
    def test_torch_unavailable_returns_cpu_fallback(self):
        snapshot = get_hardware_snapshot(torch_module=None)

        self.assertFalse(snapshot["torch"]["available"])
        self.assertEqual(snapshot["torch"]["cuda_device_count"], 0)
        self.assertEqual([device["type"] for device in snapshot["devices"]], ["cpu"])
        self.assertEqual(snapshot["devices"][0]["device"], "cpu:0")
        self.assertEqual(snapshot["default_device"], "cpu:0")

    def test_partial_backend_probe_failures_remain_nonfatal(self):
        cuda = types.SimpleNamespace(
            is_available=lambda: _raise("cuda availability failed"),
        )
        mps_backend = types.SimpleNamespace(
            is_built=lambda: _raise("mps build probe failed"),
            is_available=lambda: _raise("mps availability failed"),
        )
        mps_runtime = types.SimpleNamespace(
            is_available=lambda: _raise("mps runtime probe failed"),
        )

        snapshot = get_hardware_snapshot(
            torch_module=_fake_torch(
                cuda=cuda,
                mps_backend=mps_backend,
                mps_runtime=mps_runtime,
            )
        )

        self.assertEqual([device["type"] for device in snapshot["devices"]], ["cpu"])
        self.assertEqual(snapshot["default_device"], "cpu:0")
        errors = snapshot["torch"]["errors"]
        self.assertIn("cuda_available", errors)
        self.assertIn("mps_built", errors)
        self.assertIn("mps_available", errors)
        self.assertIn("mps_runtime_available", errors)

    def test_cuda_count_failure_does_not_claim_a_detected_accelerator(self):
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: _raise("cuda count failed"),
        )
        snapshot = get_hardware_snapshot(
            torch_module=_fake_torch(
                cuda=cuda,
                mps_backend=types.SimpleNamespace(is_built=lambda: False, is_available=lambda: False),
                mps_runtime=None,
            )
        )

        self.assertFalse(snapshot["torch"]["cuda_available"])
        self.assertEqual(snapshot["torch"]["cuda_device_count"], 0)
        self.assertEqual(snapshot["default_device"], "cpu:0")
        self.assertIn("cuda_device_count", snapshot["torch"]["errors"])

    def test_multiple_cuda_devices_survive_per_device_memory_failure(self):
        class FakeCuda:
            def is_available(self):
                return True

            def device_count(self):
                return 2

            def get_device_name(self, index):
                return f"Mock CUDA {index}"

            def get_device_properties(self, index):
                return types.SimpleNamespace(total_memory=(index + 1) * 8 * GIB)

            def mem_get_info(self, index):
                if index == 1:
                    raise RuntimeError("memory query failed")
                return 6 * GIB, 8 * GIB

            def memory_allocated(self, index):
                return (index + 1) * GIB

            def memory_reserved(self, index):
                return (index + 2) * GIB

        mps_backend = types.SimpleNamespace(
            is_built=lambda: False,
            is_available=lambda: False,
        )
        snapshot = get_hardware_snapshot(
            torch_module=_fake_torch(
                cuda=FakeCuda(),
                mps_backend=mps_backend,
                mps_runtime=None,
            )
        )

        self.assertEqual(
            [device["device"] for device in snapshot["devices"]],
            ["cuda:0", "cuda:1", "cpu:0"],
        )
        self.assertEqual(snapshot["torch"]["cuda_device_count"], 2)
        self.assertEqual(snapshot["default_device"], "cuda:0")
        self.assertEqual(snapshot["devices"][0]["vram_free"], 6 * GIB)
        second = snapshot["devices"][1]
        self.assertEqual(second["vram_total"], 16 * GIB)
        self.assertIsNone(second["vram_free"])
        self.assertEqual(second["torch_vram_total"], 16 * GIB)
        self.assertIn("memory", second["errors"])

    def test_mps_is_represented_before_cpu(self):
        cuda = types.SimpleNamespace(is_available=lambda: False)
        mps_backend = types.SimpleNamespace(
            is_built=lambda: True,
            is_available=lambda: True,
        )
        snapshot = get_hardware_snapshot(
            torch_module=_fake_torch(
                cuda=cuda,
                mps_backend=mps_backend,
                mps_runtime=None,
            )
        )

        self.assertTrue(snapshot["torch"]["mps_built"])
        self.assertTrue(snapshot["torch"]["mps_available"])
        self.assertEqual(
            [device["device"] for device in snapshot["devices"]],
            ["mps:0", "cpu:0"],
        )
        self.assertEqual(snapshot["devices"][0]["type"], "mps")
        self.assertEqual(snapshot["default_device"], "mps:0")

    def test_snapshot_schema_has_stable_system_and_devices_shape(self):
        snapshot = get_hardware_snapshot(torch_module=None)

        self.assertEqual(snapshot["schema_version"], SCHEMA_VERSION)
        self.assertEqual(
            set(snapshot),
            {"schema_version", "system", "torch", "devices", "default_device", "disk"},
        )
        self.assertEqual(
            set(snapshot["system"]),
            {
                "os",
                "os_name",
                "python_version",
                "python_executable",
                "pytorch_version",
                "argv",
                "ram_total",
                "ram_free",
                "ram_available",
                "pytorch_cuda_alloc_conf",
                "environment",
            },
        )
        self.assertIsInstance(snapshot["devices"], list)
        self.assertGreaterEqual(len(snapshot["devices"]), 1)
        self.assertEqual(
            set(snapshot["devices"][0]),
            {
                "type",
                "index",
                "device",
                "name",
                "vram_total",
                "vram_free",
                "torch_vram_total",
                "torch_vram_free",
                "torch_allocated",
                "torch_reserved",
            },
        )
        self.assertEqual(
            set(snapshot["disk"]),
            {"path", "total_bytes", "free_bytes", "used_bytes", "source", "error"},
        )
        json.dumps(snapshot)


if __name__ == "__main__":
    unittest.main()
