"""Accepted queued cancellations retain the same bounded terminal run history."""

import asyncio
import json
import os
import tempfile
import threading
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modiff.server import WebServer


class ObserverSocket:
    closed = False

    def __init__(self):
        self.messages = []

    async def prepare(self, _request):
        pass

    async def send_json(self, message):
        self.messages.append(message)

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class QueuedCancellationHistoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.state_path = self.directory / "supervisor-queue.json"
        environment = patch.dict(os.environ, {"MODIFF_SUPERVISOR_QUEUE_STATE": str(self.state_path)})
        environment.start()
        self.addCleanup(environment.stop)
        self.server = self.make_server()

    def make_server(self):
        server = WebServer(modules={}, work_dir=str(self.directory), data_dir=str(self.directory))
        server.queue_message = Mock()
        return server

    async def admit(self, label="queued", *, future=None, preview=False):
        snapshot = {"nodes": [{"id": label, "data": {"prompt": "keep café unchanged"}}], "edges": []}
        hints = {
            "clientRunId": f"run-{label}",
            "runInputHash": f"hash-{label}",
            "workflowTabId": f"workflow-{label}",
            "workflowTitle": f"Workflow {label}",
            "workflowSnapshot": snapshot,
            "workflowCanvasEpoch": 2,
            "workflowFormEpoch": 3,
        }
        graph = {"nodes": {}, "edges": [], "runtimeHints": hints}
        if preview:
            self.server.modules = {"modules.Image": {"Preview": {"params": {"images": {"display": "ui_image"}}}}}
            graph["nodes"]["preview"] = {
                "module": "modules.Image",
                "action": "Preview",
                "params": {"images": {"value": None}},
            }
        callback = Mock()
        task_id = await self.server.queue_task(callback, (graph,), future, "owner", name="Graph execution")
        return task_id, graph, callback

    def set_active(self):
        self.server.current_task = {
            "task_id": "active",
            "name": "Graph execution",
            "sid": "active-owner",
            "started_at": 1.0,
            "progress": 25,
            "current_step": 2,
            "total_steps": 8,
            "runtimeHints": {"clientRunId": "active-run", "workflowTabId": "active-workflow"},
        }
        self.server.task_graphs["active"] = {"nodes": {"active-node": {}}, "edges": []}

    async def detail(self, server, task_id):
        response = await server.get_run(SimpleNamespace(match_info={"task_id": task_id}))
        self.assertEqual(response.status, 200)
        return json.loads(response.body)

    async def assert_cancelled_detail(self, server, task_id, graph):
        detail = await self.detail(server, task_id)
        self.assertEqual(detail["task"]["status"], "cancelled")
        self.assertEqual(detail["task"]["message"], "Cancelled before execution.")
        self.assertEqual(detail["task"]["progress"], 0)
        self.assertIsNone(detail["task"].get("started_at"))
        self.assertEqual(detail["task"]["client_run_id"], graph["runtimeHints"]["clientRunId"])
        self.assertEqual(detail["workflow_id"], graph["runtimeHints"]["workflowTabId"])
        self.assertEqual(detail["workflow_title"], graph["runtimeHints"]["workflowTitle"])
        self.assertEqual(detail["workflow_snapshot"], graph["runtimeHints"]["workflowSnapshot"])
        self.assertEqual(detail["outputs"], [])
        self.assertNotIn("task", detail["task"])
        self.assertNotIn("args", detail["task"])
        self.assertNotIn("future", detail["task"])
        return detail

    async def test_delete_retains_receipt_for_fresh_observer_and_replacement_worker(self):
        self.set_active()
        active = deepcopy(self.server.current_task)
        task_id, graph, callback = await self.admit()
        next_id, _, next_callback = await self.admit("other")
        response = await self.server.delete_task(SimpleNamespace(match_info={"task_id": task_id}))
        self.assertEqual(response.status, 200)
        payload = json.loads(response.body)
        self.assertEqual(payload["task_id"], task_id)
        self.assertEqual(payload["current"]["task_id"], "active")
        self.assertEqual(set(payload["queued"]), {next_id})
        self.assertEqual(self.server.current_task, active)
        self.assertFalse(self.server.interrupt_flag)
        callback.assert_not_called()
        next_callback.assert_not_called()
        await self.assert_cancelled_detail(self.server, task_id, graph)

        cancelled = [
            call.args[0]
            for call in self.server.queue_message.call_args_list
            if call.args[0].get("type") == "task_cancelled"
        ]
        self.assertEqual(len(cancelled), 1)
        self.assertEqual(cancelled[0]["status"], "cancelled")
        self.assertEqual(cancelled[0]["client_run_id"], graph["runtimeHints"]["clientRunId"])
        self.assertEqual(cancelled[0]["recent"][0]["task_id"], task_id)
        self.assertNotIn("workflow_snapshot", cancelled[0]["recent"][0])

        # An independent handshake and queue read recover terminal truth without
        # relying on the cancelling browser's in-memory notification shelf.
        socket = ObserverSocket()
        request = SimpleNamespace(
            headers={"Origin": "http://127.0.0.1:8088"},
            host="127.0.0.1:8088",
            remote="127.0.0.1",
            query={"sid": "observer"},
        )
        with patch("modiff.server.web.WebSocketResponse", return_value=socket):
            await self.server.websocket(request)
        welcome = socket.messages[0]
        self.assertEqual(welcome["recent"][0]["status"], "cancelled")
        self.assertEqual(welcome["current"]["task_id"], "active")
        self.assertTrue(welcome["recent"][0]["has_workflow_snapshot"])
        self.assertNotIn("workflow_snapshot", welcome["recent"][0])
        queue = json.loads((await self.server.get_queue(None)).body)
        self.assertEqual(queue["recent"], welcome["recent"])

        replacement = self.make_server()
        await self.assert_cancelled_detail(replacement, task_id, graph)
        self.assertEqual(replacement.task_graphs, {})
        self.assertFalse(replacement.current_task)

    async def test_unknown_repeated_and_active_cancellation_do_not_change_history(self):
        self.set_active()
        task_id, graph, _ = await self.admit()
        await self.server.delete_task(SimpleNamespace(match_info={"task_id": task_id}))
        original_recent = deepcopy(self.server.recent_tasks)
        original_current = deepcopy(self.server.current_task)
        for target, expected in ((task_id, 404), ("missing", 404), ("active", 400)):
            with self.subTest(target=target):
                response = await self.server.delete_task(SimpleNamespace(match_info={"task_id": target}))
                self.assertEqual(response.status, expected)
                self.assertTrue(json.loads(response.body)["error"])
                self.assertEqual(self.server.recent_tasks, original_recent)
                self.assertEqual(self.server.current_task, original_current)
        await self.assert_cancelled_detail(self.server, task_id, graph)

    async def test_cancelled_queue_entry_is_skipped_without_start_or_output(self):
        future = asyncio.get_running_loop().create_future()
        task_id, graph, callback = await self.admit(future=future)
        await self.server.delete_task(SimpleNamespace(match_info={"task_id": task_id}))
        await self.server.main_queue.put(None)
        await asyncio.wait_for(self.server._main_worker(), timeout=2)
        with self.assertRaises(asyncio.CancelledError):
            await future
        callback.assert_not_called()
        self.assertTrue(self.server.main_queue.empty())
        self.assertEqual(self.server.main_queue._unfinished_tasks, 0)
        self.assertFalse(self.server.current_task)
        event_types = [call.args[0]["type"] for call in self.server.queue_message.call_args_list]
        self.assertNotIn("task_started", event_types)
        self.assertNotIn("task_completed", event_types)
        await self.assert_cancelled_detail(self.server, task_id, graph)

    async def test_cancelled_receipts_and_graphs_use_existing_thirty_run_bound(self):
        self.set_active()
        active = deepcopy(self.server.current_task)
        task_ids = []
        for index in range(35):
            task_id, _, _ = await self.admit(f"queued-{index}")
            task_ids.append(task_id)
            await self.server.delete_task(SimpleNamespace(match_info={"task_id": task_id}))
        retained = list(reversed(task_ids[-30:]))
        self.assertEqual([task["task_id"] for task in self.server.recent_tasks], retained)
        self.assertEqual(set(self.server.task_graphs), {"active", *retained})
        self.assertEqual(self.server.current_task, active)
        response = await self.server.get_run(SimpleNamespace(match_info={"task_id": task_ids[0]}))
        self.assertEqual(response.status, 404)
        persisted = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual([task["task_id"] for task in persisted["recent"]], retained)

    async def test_worker_stop_retains_queued_receipts_without_terminalizing_active_task(self):
        self.set_active()
        task_id, graph, callback = await self.admit()
        with patch.object(self.server, "_schedule_forced_restart_if_still_running", return_value=2000):
            response = await self.server.stop_execution(None)
        self.assertEqual(json.loads(response.body)["cancelled_queued_task_ids"], [task_id])
        await self.assert_cancelled_detail(self.server, task_id, graph)
        self.assertEqual(self.server.current_task["task_id"], "active")
        self.assertTrue(self.server.current_task["interrupt_requested"])
        self.assertTrue(self.server.interrupt_flag)
        self.assertEqual([entry["task_id"] for entry in self.server.recent_tasks], [task_id])
        callback.assert_not_called()

    async def test_worker_stop_with_only_queued_work_persists_cancelled_navigation(self):
        task_id, graph, callback = await self.admit()
        response = await self.server.stop_execution(None)
        payload = json.loads(response.body)
        self.assertFalse(payload["cleanup_pending"])
        self.assertFalse(self.server.interrupt_flag)
        self.assertFalse(self.server.current_task)
        await self.assert_cancelled_detail(self.make_server(), task_id, graph)
        callback.assert_not_called()

    async def test_worker_stop_settles_queued_preview_without_touching_active_preview(self):
        active_id, _, active_callback = await self.admit("active", preview=True)
        active_admission = self.server.queued_tasks.pop(active_id)
        self.server.current_task = {
            "task_id": active_id,
            "name": "Graph execution",
            "sid": "owner",
            "started_at": 1.0,
            "runtimeHints": active_admission["runtimeHints"],
        }
        task_id, _, callback = await self.admit(preview=True)
        before = self.server._read_studio_output_state()
        active_slot = next(slot for slot in before["previewSlots"].values() if slot["pendingTaskId"] == active_id)
        with patch.object(self.server, "_schedule_forced_restart_if_still_running", return_value=2000):
            await self.server.stop_execution(None)
        after = self.server._read_studio_output_state()
        cancelled_slot = next(
            slot for slot in after["previewSlots"].values() if slot["workflowTabId"] == "workflow-queued"
        )
        self.assertEqual(cancelled_slot["status"], "cancelled")
        self.assertIsNone(cancelled_slot["pendingTaskId"])
        self.assertIsNone(cancelled_slot["pendingClientRunId"])
        self.assertIn(active_slot, after["previewSlots"].values())
        self.assertEqual(self.server.current_task["task_id"], active_id)
        cancelled = next(
            call.args[0]
            for call in self.server.queue_message.call_args_list
            if call.args[0].get("type") == "task_cancelled" and call.args[0]["task_id"] == task_id
        )
        self.assertEqual(cancelled["preview_slots"], [cancelled_slot])
        self.assertEqual(cancelled["preview_state_revision"], after["revision"])
        callback.assert_not_called()
        active_callback.assert_not_called()

    async def test_replacement_reconciles_exact_terminal_pending_slots_preserving_delivered_outputs(self):
        for status, expected in (
            ("cancelled", "cancelled"),
            ("failed", "failed"),
            ("completed", "completed_without_output"),
        ):
            with self.subTest(status=status):
                self.state_path.unlink(missing_ok=True)
                self.server = self.make_server()
                task_id, _, callback = await self.admit(f"terminal-{status}", preview=True)
                unrelated_id, _, unrelated_callback = await self.admit(f"unrelated-{status}", preview=True)
                state = self.server._read_studio_output_state()
                unrelated = next(
                    slot for slot in state["previewSlots"].values() if slot["pendingTaskId"] == unrelated_id
                )
                delivered = {
                    **unrelated,
                    "workflowTabId": "delivered",
                    "currentOutputId": "delivered-output",
                    "pendingTaskId": None,
                    "pendingClientRunId": None,
                    "status": "ready",
                }
                state["previewSlots"][self.server._studio_preview_slot_key("delivered", "preview", "images")] = (
                    delivered
                )
                outputs = [{"id": "delivered-output", "taskId": "delivered-task", "createdAt": 1}]
                self.server._write_studio_output_state(outputs, state["previewSlots"], revision=state["revision"] + 1)
                self.state_path.write_text(
                    json.dumps(
                        {
                            "recent": [
                                {"task_id": task_id, "status": status},
                                {"task_id": "delivered-task", "status": status},
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                replacement = self.make_server()
                after = replacement._read_studio_output_state()
                terminal = next(
                    slot
                    for slot in after["previewSlots"].values()
                    if slot["workflowTabId"] == f"workflow-terminal-{status}"
                )
                self.assertEqual(terminal["status"], expected)
                self.assertIsNone(terminal["pendingTaskId"])
                self.assertIsNone(terminal["pendingClientRunId"])
                self.assertIn(unrelated, after["previewSlots"].values())
                self.assertIn(delivered, after["previewSlots"].values())
                self.assertEqual(after["outputs"], outputs)
                callback.assert_not_called()
                unrelated_callback.assert_not_called()

    async def test_stop_interrupts_before_history_drain_and_retains_original_task_identity(self):
        self.set_active()
        await self.admit(preview=True)
        pipeline = SimpleNamespace(_interrupt=False)
        node = SimpleNamespace(_interrupt=False, _active_pipeline=pipeline)
        self.server.node_cache = {"active-node": node}
        entered, release = threading.Event(), threading.Event()
        mark_terminal = self.server._mark_studio_preview_run_terminal

        def blocked_history(task_id, status):
            self.assertNotEqual(threading.get_ident(), threading.main_thread().ident)
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Preview cancellation did not drain.")
            return mark_terminal(task_id, status)

        stop = None
        with (
            patch.object(self.server, "_mark_studio_preview_run_terminal", side_effect=blocked_history),
            patch.object(self.server, "_schedule_forced_restart_if_still_running", return_value=2000) as schedule,
        ):
            try:
                stop = asyncio.create_task(self.server.stop_execution(None))
                self.assertTrue(await asyncio.to_thread(entered.wait, 5))
                self.assertFalse(stop.done())
                self.assertTrue(self.server.interrupt_flag)
                self.assertTrue(node._interrupt)
                self.assertTrue(pipeline._interrupt)
                schedule.assert_called_once_with("active")
                self.assertEqual(self.server.queued_tasks, {})
                await asyncio.wait_for(self.server.get_queue(None), timeout=0.25)
                # The native call may finish while the HTTP history transaction
                # drains. Its eventual Stop response still names the owner it stopped.
                self.server._record_terminal_task("cancelled")
                self.server.current_task = None
            finally:
                release.set()
                if stop is not None:
                    response = await asyncio.wait_for(stop, timeout=5)
        self.assertEqual(json.loads(response.body)["task_id"], "active")


if __name__ == "__main__":
    unittest.main()
