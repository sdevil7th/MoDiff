"""Gallery history/file locks must never run on the HTTP event loop."""

import asyncio
import json
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from modiff.server import WebServer


class Request:
    def __init__(self, payload=None, *, output_id="one", limit="1"):
        self.payload = payload
        self.match_info = {"output_id": output_id}
        self.query = {"limit": limit}

    async def json(self):
        return self.payload


class StudioHistoryResponsivenessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.server = WebServer(modules={}, work_dir=self.directory.name, data_dir=self.directory.name)
        self.server._write_studio_outputs([
            {"id": "two", "createdAt": 2, "prompt": "newer"},
            {"id": "one", "createdAt": 1, "prompt": "older"},
        ])

    async def test_slow_browser_does_not_hold_up_other_browser_completion(self):
        stalled = asyncio.Event()
        delivered = asyncio.Event()

        async def send_slow(_message):
            await stalled.wait()

        async def send_fast(_message):
            delivered.set()

        slow = SimpleNamespace(closed=False, send_json=send_slow, close=AsyncMock())
        fast = SimpleNamespace(closed=False, send_json=send_fast)
        self.server.ws_sessions = {"stalled": slow, "active": fast}
        sending = asyncio.create_task(self.server.broadcast({"type": "graph_completed"}))
        try:
            await asyncio.wait_for(delivered.wait(), timeout=0.2)
        finally:
            stalled.set()
            await sending

    async def test_gallery_patch_rejects_untrusted_fields_without_partial_writes(self):
        receipt = {'source': 'backend-execution', 'taskId': 'real-task', 'summary': {'seed': 7}}
        self.server._write_studio_outputs([{
            'id': 'one', 'taskId': 'real-task', 'nodeId': 'preview', 'attemptIndex': 0,
            'resolvedExecutionInputs': receipt, 'favorite': False,
        }])
        before = self.server._read_studio_output_state()
        for payload in (None, [], {}, {'favorite': 1}, {'favorite': 'true'},
                        {'favorite': True, 'taskId': 'forged'}, {'nodeId': 'other'},
                        {'attemptIndex': 1}, {'id': 'other'}, {'prompt': 'forged'},
                        {'resolvedExecutionInputs': {'source': 'backend-execution'}},
                        {'mediaItems': []}, {'updatedAt': 0}):
            with self.subTest(payload=payload):
                response = await self.server.studio_outputs_patch(Request(payload))
                self.assertEqual(response.status, 400, response.text)
                self.assertEqual(self.server._read_studio_output_state(), before)
        response = await self.server.studio_outputs_patch(Request({'favorite': True}))
        self.assertEqual(response.status, 200, response.text)
        updated = json.loads(response.body)['outputs'][0]
        self.assertTrue(updated['favorite'])
        for key in ('taskId', 'nodeId', 'attemptIndex', 'resolvedExecutionInputs'):
            self.assertEqual(updated[key], before['outputs'][0][key])

    async def test_gallery_patch_http_boundary_validates_json_and_keeps_legacy_favorites(self):
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        app = web.Application()
        app.router.add_patch('/studio_outputs/{output_id}', self.server.studio_outputs_patch)
        async with TestClient(TestServer(app)) as client:
            before = self.server._read_studio_output_state()
            for payload in ({'favorite': True, 'taskId': 'forged'}, {'favorite': None}):
                response = await client.patch('/studio_outputs/one', json=payload)
                self.assertEqual(response.status, 400)
                self.assertTrue((await response.json())['error'])
            response = await client.patch('/studio_outputs/one', data='{')
            self.assertEqual(response.status, 400)
            self.assertEqual(self.server._read_studio_output_state(), before)
            response = await client.patch('/studio_outputs/one', json={'favorite': True})
            self.assertEqual(response.status, 200)
            row = next(item for item in (await response.json())['outputs'] if item['id'] == 'one')
            self.assertTrue(row['favorite'])
            self.assertEqual(row['prompt'], 'older')
            response = await client.patch('/studio_outputs/absent', json={'favorite': False})
            self.assertEqual(response.status, 404)

    async def test_stalled_browser_is_closed_and_pruned_after_bounded_write(self):
        async def never_finishes(_message):
            await asyncio.Event().wait()

        slow = SimpleNamespace(closed=False, send_bytes=never_finishes, close=AsyncMock())
        fast = SimpleNamespace(closed=False, send_bytes=AsyncMock())
        self.server.ws_sessions = {"stalled": slow, "active": fast}
        with patch("modiff.server.WEBSOCKET_SEND_TIMEOUT_SECONDS", 0.01):
            await asyncio.wait_for(self.server.broadcast(b"preview"), timeout=0.5)
        fast.send_bytes.assert_awaited_once_with(b"preview")
        slow.close.assert_awaited_once_with(code=1013)
        self.assertEqual(self.server.ws_sessions, {"active": fast})

    async def test_gallery_read_and_mutations_keep_file_io_off_the_http_thread(self):
        main_thread = threading.get_ident()
        for method, request, expected_status in (
            ("studio_outputs_get", Request(), 200),
            ("studio_outputs_post", Request({"id": "three", "createdAt": 3, "prompt": "new"}), 200),
            ("studio_outputs_patch", Request({"favorite": True}), 200),
            ("studio_outputs_delete", Request(), 200),
            ("studio_outputs_patch", Request({"favorite": True}, output_id="absent"), 404),
            ("studio_outputs_delete", Request(output_id="absent"), 404),
        ):
            with self.subTest(method=method, expected_status=expected_status):
                threads = []
                original = self.server._read_studio_output_state

                def read():
                    threads.append(threading.get_ident())
                    return original()

                with patch.object(self.server, "_read_studio_output_state", side_effect=read):
                    response = await getattr(self.server, method)(request)
                self.assertEqual(response.status, expected_status, response.text)
                self.assertTrue(threads)
                self.assertNotIn(main_thread, threads)

    async def test_gallery_read_coalesces_equal_limits_but_not_different_limits(self):
        calls = []
        release = threading.Event()
        original = self.server._read_studio_output_state

        def read():
            calls.append(threading.get_ident())
            if not release.wait(2):
                raise RuntimeError("The event loop could not release the held history reader.")
            return original()

        with patch.object(self.server, "_read_studio_output_state", side_effect=read):
            one = asyncio.create_task(self.server.studio_outputs_get(Request(limit="1")))
            same = asyncio.create_task(self.server.studio_outputs_get(Request(limit="1")))
            try:
                await asyncio.sleep(0.02)
                self.assertFalse(one.done(), "Gallery read blocked the event loop instead of yielding.")
                release.set()
                first, second = await asyncio.gather(one, same)
            finally:
                release.set()
                await asyncio.gather(one, same, return_exceptions=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(first.body), json.loads(second.body))
        self.assertEqual(len(json.loads(first.body)["outputs"]), 1)
        wider = await self.server.studio_outputs_get(Request(limit="2"))
        self.assertEqual(len(json.loads(wider.body)["outputs"]), 2)

    async def test_current_preview_output_is_retained_outside_the_requested_history_limit(self):
        state = {
            "revision": 7,
            "outputs": [{"id": "new"}, {"id": "current"}, {"id": "older"}],
            "previewSlots": {"slot": {"currentOutputId": "current"}},
        }
        with patch.object(self.server, "_read_studio_output_state", return_value=state):
            response = await self.server.studio_outputs_get(SimpleNamespace(query={"limit": "1"}))
        body = json.loads(response.body)
        self.assertEqual([output["id"] for output in body["outputs"]], ["new", "current"])
        self.assertEqual(body["revision"], 7)
        self.assertEqual(body["count"], 3)

    async def test_repeated_cancellation_drains_writer_before_next_mutation(self):
        entered = threading.Event()
        release = threading.Event()
        order = []

        def slow():
            order.append("first started")
            entered.set()
            if not release.wait(2):
                raise RuntimeError("Event loop could not release the history writer.")
            order.append("first finished")

        def next_writer():
            order.append("second")

        first = asyncio.create_task(self.server._run_studio_history_mutation(slow))
        second = None
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            first.cancel()
            await asyncio.sleep(0)
            first.cancel()
            second = asyncio.create_task(self.server._run_studio_history_mutation(next_writer))
            await asyncio.sleep(0.02)
            self.assertFalse(first.done())
            self.assertFalse(second.done())
            release.set()
            results = await asyncio.gather(first, second, return_exceptions=True)
            self.assertIsInstance(results[0], asyncio.CancelledError)
            self.assertEqual(order, ["first started", "first finished", "second"])
        finally:
            release.set()
            await asyncio.gather(first, *([second] if second else []), return_exceptions=True)

    async def test_run_detail_history_and_serialization_are_off_the_http_thread(self):
        self.server.recent_tasks = [{"task_id": "task", "name": "run", "status": "completed"}]
        main_thread = threading.get_ident()
        original = self.server._run_detail_response

        def build(*args):
            self.assertNotEqual(threading.get_ident(), main_thread)
            return original(*args)

        with patch.object(self.server, "_run_detail_response", side_effect=build):
            response = await self.server.get_run(SimpleNamespace(match_info={"task_id": "task"}))
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.body)["task"]["task_id"], "task")

    async def test_live_queue_status_does_not_wait_for_gallery_commit_or_full_run_detail(self):
        self.server.current_task = {
            "task_id": "task", "name": "FLUX inpaint", "current_step": 28,
            "total_steps": 28, "phase": "previewing", "progress": 99,
        }
        entered, release = threading.Event(), threading.Event()

        def hold_gallery_writer():
            with self.server.studio_history_file_lock:
                entered.set()
                if not release.wait(2):
                    raise RuntimeError("Queue status could not proceed while Gallery commits.")

        writer = asyncio.create_task(asyncio.to_thread(hold_gallery_writer))
        detail = None
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            detail = asyncio.create_task(self.server.get_run(SimpleNamespace(match_info={"task_id": "task"})))
            await asyncio.sleep(0.02)
            self.assertFalse(detail.done())
            response = await asyncio.wait_for(self.server.get_queue(None), timeout=0.25)
            body = json.loads(response.body)
            self.assertEqual(body["current"]["task_id"], "task")
            self.assertEqual(body["current"]["current_step"], 28)
            self.assertEqual(body["current"]["phase"], "previewing")
            self.assertFalse(detail.done(), "Full history should still be waiting for its writer.")
        finally:
            release.set()
            await writer
            if detail is not None:
                response = await detail
                self.assertEqual(response.status, 200)

    async def test_failed_transaction_releases_ordering_lock_for_next_writer(self):
        def fail():
            raise OSError("read-only history")

        with self.assertRaisesRegex(OSError, "read-only history"):
            await self.server._run_studio_history_mutation(fail)
        self.assertEqual(await self.server._run_studio_history_mutation(lambda: "next"), "next")

    async def test_cancelled_queue_admission_drains_pending_write_and_removes_phantom_task(self):
        entered = threading.Event()
        release = threading.Event()
        main_thread = threading.get_ident()
        terminal = []

        def pending(_graph, _task_id):
            self.assertNotEqual(threading.get_ident(), main_thread)
            entered.set()
            if not release.wait(2):
                raise RuntimeError("Queue admission did not yield to HTTP.")
            return {"revision": 1, "previewSlots": []}

        def cancelled(task_id, status):
            self.assertNotEqual(threading.get_ident(), main_thread)
            terminal.append((task_id, status))

        with (
            patch.object(self.server, "_mark_studio_preview_slots_pending", side_effect=pending),
            patch.object(self.server, "_mark_studio_preview_run_terminal", side_effect=cancelled),
        ):
            admission = asyncio.create_task(self.server.queue_task(
                lambda: None, ({"nodes": {}, "edges": []},), None, "browser", name="Graph execution"
            ))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                task_id = next(iter(self.server.queued_tasks))
                admission.cancel()
                await asyncio.sleep(0.02)
                self.assertFalse(admission.done())
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await admission
                self.assertEqual(terminal, [(task_id, "cancelled")])
                self.assertEqual(self.server.queued_tasks, {})
                self.assertEqual(self.server.task_graphs, {})
                self.assertTrue(self.server.main_queue.empty())
            finally:
                release.set()
                await asyncio.gather(admission, return_exceptions=True)
