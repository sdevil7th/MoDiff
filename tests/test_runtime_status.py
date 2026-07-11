import copy
import io
import json
import mimetypes
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault(
    "aiohttp_cors",
    types.SimpleNamespace(
        setup=lambda *args, **kwargs: None,
        ResourceOptions=lambda *args, **kwargs: None,
    ),
)
sys.modules.setdefault(
    "nanoid",
    types.SimpleNamespace(generate=lambda size=12: "test-id"),
)

from modiff import preflight  # noqa: E402
from modiff.server import WebServer  # noqa: E402
from aiohttp.web_fileresponse import CONTENT_TYPES as AIOHTTP_CONTENT_TYPES  # noqa: E402


GIB = 1024**3


def hardware_snapshot(*, ram_total=32 * GIB, cuda=True):
    devices = []
    if cuda:
        devices.append({
            "type": "cuda",
            "index": 0,
            "device": "cuda:0",
            "name": "Mock CUDA",
            "vram_total": 16 * GIB,
            "vram_free": 12 * GIB,
            "torch_vram_total": 16 * GIB,
            "torch_vram_free": 12 * GIB,
            "torch_allocated": 2 * GIB,
            "torch_reserved": 4 * GIB,
        })
    devices.append({
        "type": "cpu",
        "index": 0,
        "device": "cpu:0",
        "name": "Mock CPU",
        "vram_total": None,
        "vram_free": None,
        "torch_vram_total": None,
        "torch_vram_free": None,
        "torch_allocated": None,
        "torch_reserved": None,
    })
    return {
        "schema_version": 1,
        "system": {
            "os": "unit-test",
            "os_name": "nt",
            "python_version": "3.12-test",
            "python_executable": "python",
            "pytorch_version": "2.test",
            "argv": ["test"],
            "ram_total": ram_total,
            "ram_free": ram_total // 2,
            "ram_available": ram_total // 2,
            "pytorch_cuda_alloc_conf": "expandable_segments:True",
            "environment": {},
        },
        "torch": {
            "available": True,
            "version": "2.test",
            "cuda_available": cuda,
            "cuda_device_count": 1 if cuda else 0,
            "mps_built": False,
            "mps_available": False,
            "cudnn_version": 9000 if cuda else None,
            "cudnn_deterministic": False if cuda else None,
            "cudnn_benchmark": False if cuda else None,
            "deterministic_algorithms": False,
        },
        "devices": devices,
        "default_device": "cuda:0" if cuda else "cpu:0",
        "disk": {
            "path": "unit-test",
            "total_bytes": 256 * GIB,
            "free_bytes": 128 * GIB,
            "used_bytes": 128 * GIB,
            "source": "unit-test",
            "error": None,
        },
    }


def available_package(module_name, distribution_name=None):
    return {
        "available": True,
        "module": module_name,
        "distribution": distribution_name or module_name,
        "version": "unit-test",
    }


class RuntimeStatusTests(unittest.IsolatedAsyncioTestCase):
    def test_webp_assets_use_browser_image_content_type(self):
        self.assertEqual(mimetypes.guess_type("gallery-image.webp")[0], "image/webp")
        self.assertEqual(AIOHTTP_CONTENT_TYPES.guess_type("gallery-image.webp")[0], "image/webp")

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.server = WebServer(modules={"unit": {}}, work_dir=self.temp_dir.name, data_dir=self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    async def test_runtime_status_preserves_existing_fields_and_adds_hardware(self):
        snapshot = hardware_snapshot()
        self.server._package_status = available_package

        with patch("modiff.server.get_hardware_snapshot", return_value=copy.deepcopy(snapshot)) as get_snapshot:
            response = await self.server.runtime_status(None)

        payload = json.loads(response.text)
        self.assertTrue({
            "error",
            "ready",
            "instance",
            "server",
            "python",
            "config",
            "packages",
            "missing_required_packages",
            "modules",
            "queue",
        }.issubset(payload))
        self.assertEqual(payload["hardware"], snapshot)
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["packages"]["torch"]["cuda_device_name"], "Mock CUDA")
        self.assertEqual(payload["packages"]["torch"]["cuda_memory_free_bytes"], 12 * GIB)
        get_snapshot.assert_called_once_with(self.server.data_dir)

    async def test_system_stats_route_returns_normalized_snapshot(self):
        snapshot = hardware_snapshot()
        route_paths = {route.resource.canonical for route in self.server.app.router.routes()}
        self.assertIn("/system_stats", route_paths)

        with patch("modiff.server.get_hardware_snapshot", return_value=copy.deepcopy(snapshot)):
            response = await self.server.system_stats(None)

        payload = json.loads(response.text)
        self.assertEqual(payload["system"], snapshot["system"])
        self.assertEqual(payload["devices"], snapshot["devices"])
        self.assertEqual(payload["default_device"], "cuda:0")

    def test_runtime_fingerprint_exposes_hardware_without_hashing_ram_or_disk(self):
        first_snapshot = hardware_snapshot(ram_total=32 * GIB, cuda=False)
        second_snapshot = hardware_snapshot(ram_total=64 * GIB, cuda=False)
        second_snapshot["disk"]["free_bytes"] = 64 * GIB

        with patch(
            "modiff.server.get_hardware_snapshot",
            side_effect=[copy.deepcopy(first_snapshot), copy.deepcopy(second_snapshot)],
        ):
            first = self.server._runtime_fingerprint()
            second = self.server._runtime_fingerprint()

        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(first["hardware"]["system"]["ram_total"], 32 * GIB)
        self.assertEqual(second["hardware"]["system"]["ram_total"], 64 * GIB)
        self.assertTrue({"packages", "torch", "work_dir", "data_dir", "hardware"}.issubset(first))
        self.assertFalse(first["torch"]["cuda_available"])


class PreflightHardwareTests(unittest.TestCase):
    def test_report_adds_hardware_and_preserves_torch_human_summary(self):
        snapshot = hardware_snapshot()

        def package_status(module_name, distribution_name, import_check=True):
            return {
                "module": module_name,
                "distribution": distribution_name,
                "available": True,
                "importChecked": import_check,
                "import_ms": 1,
                "version": "unit-test",
            }

        args = SimpleNamespace(check_port=65534, full=False)
        with (
            patch("modiff.preflight.package_status", side_effect=package_status),
            patch("modiff.preflight.get_hardware_snapshot", return_value=copy.deepcopy(snapshot)),
            patch("modiff.preflight.port_in_use", return_value=False),
        ):
            report = preflight.build_report(args)

        torch_status = next(item for item in report["packages"]["required"] if item["module"] == "torch")
        self.assertEqual(report["hardware"], snapshot)
        self.assertEqual(torch_status["cuda_device_name"], "Mock CUDA")
        self.assertTrue(torch_status["cuda_available"])

        output = io.StringIO()
        with redirect_stdout(output):
            preflight.print_human(report)
        self.assertIn("Torch: unit-test CUDA available (Mock CUDA); MPS not available", output.getvalue())


if __name__ == "__main__":
    unittest.main()
