import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modiff.server import WebServer


class GraphQueueAcknowledgementTests(unittest.IsolatedAsyncioTestCase):
    async def test_unsupervised_server_cannot_write_a_supervisor_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MODIFF_SUPERVISOR_QUEUE_STATE", None)
                server = WebServer(modules={}, work_dir=directory, data_dir=directory)
                server.queue_message = lambda *args, **kwargs: None

            await server.queue_task(lambda: None, (), None, "test-session", name="Graph execution")

            self.assertIsNone(server._supervisor_queue_state_path)
            self.assertFalse((Path(directory) / "runtime" / "supervisor-queue.json").exists())

    async def test_replacement_worker_retains_supervisor_terminal_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "supervisor-queue.json"
            state_path.write_text(
                json.dumps(
                    {
                        "workerPid": 1234,
                        "queued": {},
                        "current": None,
                        "recent": [
                            {
                                "task_id": "cancelled-task",
                                "name": "Graph execution",
                                "status": "cancelled",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"MODIFF_SUPERVISOR_QUEUE_STATE": str(state_path)}):
                server = WebServer(modules={}, work_dir=directory, data_dir=directory)

            self.assertEqual(server.recent_tasks[0]["task_id"], "cancelled-task")
            self.assertEqual(server.recent_tasks[0]["status"], "cancelled")

    async def test_terminal_queue_receipt_retains_resource_measurement(self):
        server = WebServer(modules={})
        server.current_task = {
            "task_id": "qualified-task",
            "name": "Graph execution",
            "sid": "session",
            "started_at": 1.0,
            "runtimeFingerprint": "runtime-lock",
            "resourceCandidateId": "qwen-native",
            "runtimeMeasurement": {
                "elapsedSeconds": 12.5,
                "peakAllocatedBytes": 8_589_934_592,
            },
        }

        current = server._current_task_snapshot()
        terminal = server._record_terminal_task("completed")

        self.assertEqual(current["resourceCandidateId"], "qwen-native")
        self.assertEqual(current["runtimeMeasurement"]["peakAllocatedBytes"], 8_589_934_592)
        self.assertEqual(terminal["runtimeFingerprint"], "runtime-lock")
        self.assertEqual(terminal["resourceCandidateId"], "qwen-native")
        self.assertEqual(terminal["runtimeMeasurement"]["elapsedSeconds"], 12.5)

    async def test_graph_execution_leaves_time_for_queue_ack_before_model_work(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        server = WebServer(modules={}, work_dir=directory.name, data_dir=directory.name)
        server.loop = asyncio.get_running_loop()
        server.queue_message = lambda *args, **kwargs: None
        executor_started = asyncio.Event()
        started_at = server.loop.time()

        async def fake_run_executor(callback, *, serialize_model_io=False, on_start=None):
            self.assertTrue(serialize_model_io)
            self.assertGreaterEqual(server.loop.time() - started_at, 0.04)
            if on_start is not None:
                on_start()
            executor_started.set()
            return callback()

        server.serialize_model_io = True
        server._run_executor_callback = fake_run_executor
        worker = asyncio.create_task(server._main_worker())
        try:
            await server.queue_task(lambda: None, (), None, 'test-session', name='Graph execution')
            await asyncio.wait_for(executor_started.wait(), timeout=1)
        finally:
            server._shutdown_event.set()
            await asyncio.wait_for(worker, timeout=2)


if __name__ == '__main__':
    unittest.main()
