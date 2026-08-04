import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modiff import hardware  # noqa: E402


GIB = 1024**3


def _fake_torch(cuda=None):
    module = types.ModuleType("torch")
    module.cuda = cuda or types.SimpleNamespace(is_available=lambda: False)
    module.float32 = object()
    module.float16 = object()
    module.bfloat16 = object()
    module.float8_e4m3fn = object()
    module.Tensor = type("Tensor", (), {})
    module.channels_last = object()
    return module


def _load_torch_utils(*, torch_module, devices=None, probe_error=None):
    module_path = ROOT / "utils" / "torch_utils.py"
    spec = importlib.util.spec_from_file_location("_isolated_test_torch_utils", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to create an isolated torch_utils module spec")
    module = importlib.util.module_from_spec(spec)

    if probe_error is not None:
        probe_patch = mock.patch.object(
            hardware,
            "get_normalized_devices",
            side_effect=probe_error,
        )
    else:
        probe_patch = mock.patch.object(
            hardware,
            "get_normalized_devices",
            return_value=list(devices or []),
        )

    with mock.patch.dict(sys.modules, {"torch": torch_module}), probe_patch:
        spec.loader.exec_module(module)
    return module


class TorchUtilsCompatibilityTests(unittest.TestCase):
    def test_import_is_safe_when_central_device_probe_raises(self):
        with self.assertLogs("modiff", level="WARNING") as logs:
            module = _load_torch_utils(
                torch_module=_fake_torch(),
                probe_error=RuntimeError("hardware probe failed"),
            )

        self.assertIn("using the CPU fallback", "\n".join(logs.output))
        self.assertEqual(
            module.DEVICE_LIST,
            {
                "cpu:0": {
                    "arch": "cpu",
                    "name": "CPU (0)",
                    "label": ["cpu:0"],
                    "total_memory": 0,
                    "index": 0,
                }
            },
        )
        self.assertEqual(module.DEFAULT_DEVICE, "cpu:0")
        self.assertEqual(module.CPU_DEVICE, "cpu:0")
        self.assertFalse(module.IS_CUDA)

    def test_legacy_device_order_schema_and_cpu_retention(self):
        normalized_devices = [
            {
                "type": "mps",
                "index": 0,
                "device": "mps:0",
                "name": "Apple Metal Performance Shaders",
                "vram_total": None,
            },
            {
                "type": "cuda",
                "index": 1,
                "device": "cuda:1",
                "name": "Second GPU",
                "vram_total": 24 * GIB,
            },
            {
                "type": "cuda",
                "index": 0,
                "device": "cuda:0",
                "name": "First GPU",
                "vram_total": 16 * GIB,
            },
        ]
        module = _load_torch_utils(
            torch_module=_fake_torch(),
            devices=normalized_devices,
        )

        self.assertEqual(
            list(module.DEVICE_LIST),
            ["cuda:0", "cuda:1", "mps:0", "cpu:0"],
        )
        self.assertEqual(module.DEFAULT_DEVICE, "cuda:0")
        self.assertEqual(module.CPU_DEVICE, "cpu:0")
        self.assertTrue(module.IS_CUDA)
        self.assertEqual(
            module.DEVICE_LIST["cuda:0"],
            {
                "arch": "cuda",
                "backend": "cuda",
                "vendor": None,
                "architecture": None,
                "memory_kind": "dedicated",
                "name": "First GPU 16.00GB (0)",
                "label": ["cuda:0"],
                "total_memory": 16 * GIB,
                "index": 0,
            },
        )
        self.assertEqual(
            set(module.DEVICE_LIST["mps:0"]),
            {
                "arch",
                "backend",
                "vendor",
                "architecture",
                "memory_kind",
                "name",
                "label",
                "total_memory",
                "index",
            },
        )
        self.assertEqual(
            module.DEVICE_LIST["cpu:0"],
            {
                "arch": "cpu",
                "name": "CPU (0)",
                "label": ["cpu:0"],
                "total_memory": 0,
                "index": 0,
            },
        )

    def test_memory_helper_failures_are_nonfatal(self):
        class FailingMemoryCuda:
            def is_available(self):
                return True

            def memory_stats(self):
                raise RuntimeError("memory stats failed")

            def reset_peak_memory_stats(self):
                raise RuntimeError("reset failed")

        module = _load_torch_utils(
            torch_module=_fake_torch(cuda=FailingMemoryCuda()),
            devices=[],
        )

        self.assertEqual(module.get_memory_stats(), {})
        module.reset_memory_stats()

    def test_malformed_memory_stats_are_nonfatal(self):
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            memory_stats=lambda: None,
            reset_peak_memory_stats=lambda: None,
        )
        module = _load_torch_utils(
            torch_module=_fake_torch(cuda=cuda),
            devices=[],
        )

        self.assertEqual(module.get_memory_stats(), {})


if __name__ == "__main__":
    unittest.main()
