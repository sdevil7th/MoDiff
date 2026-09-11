"""Active-run telemetry must never contend with the model's Torch allocator."""

import asyncio
import copy
import threading
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modiff.server import WebServer
from modiff.runtime_telemetry import active_accelerator_snapshot


class ActiveRuntimeTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.server = WebServer(modules={}, work_dir=self.directory.name, data_dir=self.directory.name)
        self.server.current_task = {"task_id": "running", "name": "Graph execution", "progress": 60}
        self.hardware = {"devices": [{
            "device": "cuda:0", "type": "cuda", "index": 0, "name": "AMD GPU",
            "backend": "rocm", "vendor": "amd", "memory_kind": "shared",
            "planning_memory_total": 2000, "planning_memory_free": 1900,
            "dedicated_memory_total": 2000, "shared_memory_total": 10000,
            "torch_allocated": 30, "torch_reserved": 40,
        }]}
        self.server._last_runtime_fingerprint = {"hardware": self.hardware}

    def test_active_snapshot_does_not_probe_torch_or_relabel_old_allocator_values(self):
        before = copy.deepcopy(self.hardware)
        with (
            patch("modiff.server.import_module", side_effect=AssertionError("must not import Torch")),
            patch("modiff.server.get_hardware_snapshot", side_effect=AssertionError("must not probe hardware")),
            patch("modiff.runtime_telemetry.Path", return_value=Path(self.directory.name)),
        ):
            snapshot = self.server._collect_runtime_resource_snapshot()
        gpu = snapshot["accelerators"][0]
        self.assertEqual(gpu["device"], "cuda:0")
        self.assertEqual(gpu["memoryKind"], "shared")
        self.assertEqual(gpu["memoryTotalBytes"], 2000)
        for key in ("allocatedBytes", "reservedBytes", "peakAllocatedBytes", "peakReservedBytes", "memoryFreeBytes", "memoryUsedBytes"):
            self.assertIsNone(gpu[key], key)
        self.assertEqual(gpu["allocatorStatsStatus"], "paused_during_execution")
        self.assertEqual(snapshot["currentRun"]["taskId"], "running")
        self.assertEqual(self.hardware, before)
        self.assertFalse(any("accelerator telemetry" in error for error in snapshot["errors"]))

    def test_active_snapshot_without_startup_hardware_does_not_fall_back_to_live_torch(self):
        self.server._last_runtime_fingerprint = None
        with (
            patch("modiff.server.import_module", side_effect=AssertionError("must not import Torch")),
            patch("modiff.server.get_hardware_snapshot", side_effect=AssertionError("must not probe hardware")),
        ):
            snapshot = self.server._collect_runtime_resource_snapshot()
        self.assertEqual(snapshot["accelerators"], [])
        self.assertIn("Accelerator topology unavailable during execution.", snapshot["errors"])

    def test_direct_gpu_diagnostic_during_run_does_not_probe_allocator(self):
        with patch("modiff.server.import_module", side_effect=AssertionError("must not import Torch")):
            snapshot = self.server._cuda_memory_snapshot()
        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["devices"][0]["name"], "AMD GPU")
        self.assertIsNone(snapshot["devices"][0]["allocated_bytes"])
        self.assertEqual(snapshot["allocator_stats_status"], "paused_during_execution")

    def test_partial_server_diagnostics_do_not_mask_the_original_execution_error(self):
        server = object.__new__(WebServer)
        with patch("modiff.server.import_module", side_effect=AssertionError("must not import Torch")):
            snapshot = server._cuda_memory_snapshot()
        self.assertFalse(snapshot["available"])
        self.assertIn("not initialized", snapshot["error"])

    def test_resource_cache_is_invalidated_on_start_finish_and_task_change(self):
        with patch.object(self.server, "_collect_runtime_resource_snapshot", return_value={}) as collect:
            for task in (None, {"task_id": "one"}, {"task_id": "two"}, None):
                self.server.current_task = task
                self.server._runtime_resource_snapshot(max_age_seconds=1000)
                self.server._runtime_resource_snapshot(max_age_seconds=1000)
        self.assertEqual(collect.call_count, 4)

    def test_single_amd_device_uses_live_os_memory_without_importing_torch(self):
        root = Path(self.directory.name)
        card = root / "card0" / "device"
        card.mkdir(parents=True)
        for key, value in {"vendor": "0x1002", "mem_info_vram_total": "2000", "mem_info_vram_used": "300",
                           "mem_info_gtt_total": "10000", "mem_info_gtt_used": "1234"}.items():
            (card / key).write_text(value, encoding="utf-8")
        with patch("modiff.runtime_telemetry.Path", return_value=root):
            snapshot = active_accelerator_snapshot(self.hardware)[0]
        self.assertEqual(snapshot["memoryFreeBytes"], 1700)
        self.assertEqual(snapshot["memoryUsedBytes"], 300)
        self.assertEqual(snapshot["sharedMemoryFreeBytes"], 8766)
        self.assertEqual(snapshot["memorySource"], "sysfs")
        self.assertIsNone(snapshot["allocatedBytes"])

    def test_ambiguous_multi_gpu_mapping_does_not_invent_device_memory(self):
        root = Path(self.directory.name)
        for name in ("card0", "card1"):
            card = root / name / "device"
            card.mkdir(parents=True)
            (card / "vendor").write_text("0x1002", encoding="utf-8")
            (card / "mem_info_vram_total").write_text("2000", encoding="utf-8")
            (card / "mem_info_vram_used").write_text("300", encoding="utf-8")
        with patch("modiff.runtime_telemetry.Path", return_value=root):
            snapshot = active_accelerator_snapshot(self.hardware)[0]
        self.assertIsNone(snapshot["memoryFreeBytes"])
        self.assertIsNone(snapshot["memorySource"])

    def test_other_accelerators_keep_topology_without_dynamic_torch_probes(self):
        for backend, kind in (("cuda", "cuda"), ("xpu", "xpu"), ("mps", "mps")):
            with self.subTest(backend=backend):
                hardware = copy.deepcopy(self.hardware)
                hardware["devices"][0].update(type=kind, device=f"{kind}:0", backend=backend)
                snapshot = active_accelerator_snapshot(hardware)[0]
                self.assertEqual(snapshot["backend"], backend)
                self.assertIsNone(snapshot["allocatedBytes"])
                self.assertIsNone(snapshot["memoryFreeBytes"])


class ResourceProbeHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_gpu_process_diagnostics_run_off_the_aiohttp_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            server = WebServer(modules={}, work_dir=directory, data_dir=directory)
            main_thread = threading.get_ident()

            def probe():
                self.assertNotEqual(threading.get_ident(), main_thread)
                return {"available": False}

            with (
                patch.object(server, "_cuda_memory_snapshot", side_effect=probe),
                patch.object(server, "_gpu_process_snapshot", side_effect=probe),
            ):
                response = await server.runtime_gpu_processes(None)
            self.assertEqual(response.status, 200)

    async def test_model_callback_waits_for_already_started_idle_probe_without_blocking_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            server = WebServer(modules={}, work_dir=directory, data_dir=directory)
            server.loop = asyncio.get_running_loop()
            entered = threading.Event()
            release = threading.Event()
            called = threading.Event()

            def idle_probe():
                with server._runtime_resource_lock:
                    entered.set()
                    release.wait(2)

            probe = threading.Thread(target=idle_probe)
            probe.start()
            await asyncio.to_thread(entered.wait, 1)
            pending = asyncio.create_task(server._run_executor_callback(called.set))
            try:
                await asyncio.sleep(0.03)
                self.assertFalse(called.is_set())
            finally:
                release.set()
                await asyncio.wait_for(pending, 2)
                await asyncio.to_thread(probe.join, 2)
            self.assertTrue(called.is_set())


if __name__ == "__main__":
    unittest.main()
