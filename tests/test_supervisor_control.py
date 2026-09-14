import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import Mock, patch

import modiff.supervisor_control as supervisor_control

from modiff.supervisor_control import (
    SupervisorControlServer,
    SupervisorController,
    _allowed_browser_origin,
    compact_task_history,
    _read_json,
    _replace_queue_snapshot,
    _write_json_atomic,
)


class SupervisorControlTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "requires Windows file sharing")
    def test_queue_publication_recovers_after_a_concurrent_reader_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "supervisor-queue.json"
            previous = {"workerPid": 4242, "current": {"task_id": "active", "progress": 1}, "label": "图像"}
            updated = {**previous, "current": {"task_id": "active", "progress": 2}}
            completed = {**previous, "current": None, "recent": [{"task_id": "active", "status": "completed"}]}
            _write_json_atomic(state_path, previous)
            real_replace = os.replace
            for next_snapshot in (updated, completed):
                contended = threading.Event()

                def observe_replace(*args):
                    try:
                        return real_replace(*args)
                    except PermissionError:
                        contended.set()
                        raise

                # The worker must keep its pending update while the supervisor
                # finishes a real read, then publish without another UI event.
                reader = state_path.open("r", encoding="utf-8")
                with ThreadPoolExecutor(max_workers=1) as executor:
                    with patch.object(supervisor_control.os, "replace", side_effect=observe_replace):
                        future = executor.submit(_write_json_atomic, state_path, next_snapshot)
                        try:
                            self.assertTrue(contended.wait(timeout=2))
                            self.assertEqual(json.load(reader), previous)
                        finally:
                            reader.close()
                        future.result(timeout=2)
                self.assertEqual(_read_json(state_path), next_snapshot)
                previous = next_snapshot

    def test_queue_replacement_retries_are_bounded_and_preserve_other_errors(self):
        source, target = Path("snapshot.tmp"), Path("snapshot.json")
        for platform, error, winerror, attempts in (
            ("nt", PermissionError("blocked"), 5, 5),
            ("nt", PermissionError("shared"), 32, 5),
            ("nt", PermissionError("privilege"), 1314, 1),
            ("posix", PermissionError("denied"), 5, 1),
            ("nt", FileNotFoundError("missing"), 2, 1),
        ):
            error.winerror = winerror
            with (
                self.subTest(platform=platform, error=type(error).__name__, winerror=winerror),
                patch.object(supervisor_control.os, "name", platform),
                patch.object(supervisor_control.os, "replace", side_effect=error) as replace,
                patch.object(supervisor_control.time, "sleep") as sleep,
            ):
                with self.assertRaises(type(error)) as raised:
                    _replace_queue_snapshot(source, target)
                self.assertIs(raised.exception, error)
                self.assertEqual(replace.call_count, attempts)
                self.assertEqual(sleep.call_count, attempts - 1)

    def test_unavailable_or_invalid_queue_snapshots_remain_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "supervisor-queue.json"
            self.assertEqual(_read_json(state_path), {})
            for value in (b"", b"{incomplete", b"[]", b"null", b"\xff"):
                with self.subTest(value=value):
                    state_path.write_bytes(value)
                    self.assertEqual(_read_json(state_path), {})
                    # A failed decode must release its file handle as well.
                    _write_json_atomic(state_path, {"current": None})
                    self.assertEqual(_read_json(state_path), {"current": None})

    def test_compaction_does_not_invent_execution_identity_from_resource_only_receipt(self):
        task = {"runtimeFingerprint": {"resourceFingerprint": "sha256:resource"}}
        self.assertIsNone(compact_task_history([task])[0]["runtimeFingerprint"])
        self.assertEqual(task["runtimeFingerprint"], {"resourceFingerprint": "sha256:resource"})
        self.assertEqual(
            compact_task_history([{"runtimeFingerprint": "sha256:execution"}])[0]["runtimeFingerprint"],
            "sha256:execution",
        )

    def test_browser_origin_policy_allows_only_local_origins(self):
        self.assertTrue(_allowed_browser_origin("http://localhost:5173"))
        self.assertTrue(_allowed_browser_origin("http://127.0.0.1:8088"))
        self.assertFalse(_allowed_browser_origin("https://attacker.example"))
        self.assertFalse(_allowed_browser_origin("null"))

    def test_control_server_rejects_get_stop_and_cross_site_post(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = SupervisorController(Path(directory) / "supervisor-queue.json")
            worker = Mock(pid=4242)
            worker.poll.return_value = None
            controller.set_worker(worker)
            server = SupervisorControlServer(controller, "127.0.0.1", 0)
            server.start()
            port = server.server.server_address[1]
            try:
                connection = HTTPConnection("127.0.0.1", port, timeout=2)
                connection.request("GET", "/stop")
                self.assertEqual(connection.getresponse().status, 404)
                connection.close()

                connection = HTTPConnection("127.0.0.1", port, timeout=2)
                connection.request("POST", "/stop", headers={"Origin": "https://attacker.example"})
                response = connection.getresponse()
                self.assertEqual(response.status, 403)
                self.assertIsNone(response.getheader("Access-Control-Allow-Origin"))
                connection.close()

                connection = HTTPConnection("127.0.0.1", port, timeout=2)
                connection.request("GET", "/queue", headers={"Host": "attacker.example"})
                response = connection.getresponse()
                self.assertEqual(response.status, 403)
                self.assertIsNone(response.getheader("Access-Control-Allow-Origin"))
                connection.close()
            finally:
                server.close()
            worker.kill.assert_not_called()

    def test_stop_kills_worker_and_cancels_current_and_queued_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "supervisor-queue.json"
            state_path.write_text(
                json.dumps(
                    {
                        "workerPid": 4242,
                        "current": {"task_id": "active", "name": "Graph execution"},
                        "queued": {"next": {"task_id": "next", "name": "Graph execution"}},
                        "recent": [
                            {
                                "task_id": "prior",
                                "status": "completed",
                                "workflow_snapshot": {"nodes": [{"id": "recoverable"}], "edges": []},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            worker = Mock(pid=4242)
            worker.poll.return_value = None
            controller = SupervisorController(state_path)
            controller.set_worker(worker)

            status, payload = controller.stop()

            self.assertEqual(status, 200)
            self.assertFalse(payload["error"])
            self.assertEqual(payload["task_id"], "active")
            self.assertEqual(payload["cancelled_queued_task_ids"], ["next"])
            self.assertTrue(payload["backend_restart"])
            worker.kill.assert_called_once_with()
            persisted = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertIsNone(persisted["current"])
            self.assertEqual(persisted["queued"], {})
            self.assertEqual(
                [task["task_id"] for task in persisted["recent"][:2]],
                ["active", "next"],
            )
            self.assertTrue(all(task["status"] == "cancelled" for task in persisted["recent"][:2]))
            self.assertEqual(
                persisted["recent"][2]["workflow_snapshot"]["nodes"][0]["id"],
                "recoverable",
            )

    def test_queue_rejects_state_from_a_replaced_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "supervisor-queue.json"
            state_path.write_text(
                json.dumps(
                    {
                        "workerPid": 1111,
                        "current": {"task_id": "stale"},
                        "queued": {},
                        "recent": [{"task_id": "done", "status": "completed"}],
                    }
                ),
                encoding="utf-8",
            )
            worker = Mock(pid=2222)
            worker.poll.return_value = None
            controller = SupervisorController(state_path)
            controller.set_worker(worker)

            queue = controller.queue()

            self.assertIsNone(queue["current"])
            self.assertEqual(queue["queued"], {})
            self.assertEqual(queue["recent"][0]["task_id"], "done")

    def test_queue_compacts_completed_workflow_snapshots_for_polling(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "supervisor-queue.json"
            state_path.write_text(
                json.dumps(
                    {
                        "workerPid": 4242,
                        "current": None,
                        "queued": {},
                        "recent": [
                            {
                                "task_id": "done",
                                "status": "completed",
                                "runtimeFingerprint": {
                                    "fingerprint": "sha256:execution",
                                    "resourceFingerprint": "sha256:resource",
                                    "hardware": {"large": "payload"},
                                },
                                "workflow_snapshot": {"nodes": [{"id": "large"}], "edges": []},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            worker = Mock(pid=4242)
            worker.poll.return_value = None
            controller = SupervisorController(state_path)
            controller.set_worker(worker)

            task = controller.queue()["recent"][0]

            self.assertNotIn("workflow_snapshot", task)
            self.assertTrue(task["has_workflow_snapshot"])
            # Runtime provenance and resource-cache identity are distinct. The
            # compact receipt must corroborate graph_completed, not substitute
            # its hardware/resource key for the executing runtime identity.
            self.assertEqual(task["runtimeFingerprint"], "sha256:execution")

    def test_stop_restarts_worker_when_its_snapshot_was_replaced_or_corrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "supervisor-queue.json"
            state_path.write_text(
                json.dumps(
                    {
                        "workerPid": 1111,
                        "current": None,
                        "queued": {},
                        "recent": [{"task_id": "done", "status": "completed"}],
                    }
                ),
                encoding="utf-8",
            )
            worker = Mock(pid=2222)
            worker.poll.return_value = None
            controller = SupervisorController(state_path)
            controller.set_worker(worker)

            status, payload = controller.stop()

            self.assertEqual(status, 200)
            self.assertTrue(payload["snapshot_recovery"])
            self.assertTrue(payload["backend_restart"])
            worker.kill.assert_called_once_with()
            persisted = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["workerPid"], 2222)
            self.assertIsNone(persisted["current"])
            self.assertEqual(persisted["recent"][0]["task_id"], "done")

    def test_unexpected_worker_exit_preserves_active_failure_and_cancels_queued_work(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "supervisor-queue.json"
            state_path.write_text(
                json.dumps(
                    {
                        "workerPid": 4242,
                        "current": {
                            "task_id": "active",
                            "status": "running",
                            "phase": "denoising",
                            "workflow_snapshot": {"nodes": [{"id": "recoverable"}], "edges": []},
                        },
                        "queued": {"next": {"task_id": "next", "status": "queued"}},
                        "recent": [{"task_id": "prior", "status": "completed"}],
                    }
                ),
                encoding="utf-8",
            )
            controller = SupervisorController(state_path)

            self.assertTrue(controller.reconcile_interrupted_worker(worker_pid=4242, return_code=-9))

            persisted = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertIsNone(persisted["current"])
            self.assertEqual(persisted["queued"], {})
            self.assertEqual([task["task_id"] for task in persisted["recent"][:3]], ["active", "next", "prior"])
            self.assertEqual(persisted["recent"][0]["status"], "failed")
            self.assertEqual(persisted["recent"][0]["error_code"], "backend_worker_exited")
            self.assertTrue(persisted["recent"][0]["backend_restart"])
            self.assertEqual(
                persisted["recent"][0]["workflow_snapshot"]["nodes"][0]["id"],
                "recoverable",
            )
            self.assertEqual(persisted["recent"][1]["status"], "cancelled")

    def test_worker_exit_reconciliation_rejects_an_unrelated_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "supervisor-queue.json"
            state_path.write_text(
                json.dumps(
                    {
                        "workerPid": 1111,
                        "current": {"task_id": "other-worker"},
                        "queued": {},
                        "recent": [],
                    }
                ),
                encoding="utf-8",
            )
            controller = SupervisorController(state_path)

            self.assertFalse(controller.reconcile_interrupted_worker(worker_pid=2222, return_code=-9))
            self.assertEqual(
                json.loads(state_path.read_text(encoding="utf-8"))["current"]["task_id"],
                "other-worker",
            )


if __name__ == "__main__":
    unittest.main()
