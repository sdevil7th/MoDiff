import json
import tempfile
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import Mock

from modiff.supervisor_control import SupervisorControlServer, SupervisorController, _allowed_browser_origin


class SupervisorControlTests(unittest.TestCase):
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
            self.assertEqual(task["runtimeFingerprint"], "sha256:resource")

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


if __name__ == "__main__":
    unittest.main()
