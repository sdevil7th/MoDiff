import asyncio
import json
import unittest
from unittest import mock

from modiff.server import WebServer


class FakeRequest:
    def __init__(self, **query):
        self.query = query


class HuggingFaceDownloadConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _space_plan(**overrides):
        return {
            "repoId": "unit/exact-model",
            "revision": "a" * 40,
            "snapshotCommit": "a" * 40,
            "selectionLimited": False,
            "requestedFiles": [],
            "totalBytes": 100,
            "completedBytes": 0,
            "remainingBytes": 100,
            "totalFileCount": 2,
            "sizeKnown": True,
            "planError": None,
            "cacheRoot": "/app-cache",
            "freeBytes": 1000,
            "totalFilesystemBytes": 2000,
            "reserveBytes": 200,
            "fits": True,
            **overrides,
        }

    async def test_app_download_forwards_exact_commit_to_hub_snapshot(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        revision = "a" * 40
        entry = {
            "task_id": "download-task",
            "sids": set(),
            "started_at": 1.0,
            "repair": False,
            "repair_source_repo_id": None,
            "requested_files": [],
            "revision": revision,
        }

        async def run_callback(callback, **_kwargs):
            return callback()

        observed_reservations = []

        def fake_download(_repo_id, progress_cb, *_args):
            progress_cb({"progress": 0.75, "remaining_bytes": 25})
            observed_reservations.append(entry["reserved_bytes"])
            return {"repo_id": "unit/exact-model", "complete": True}

        with (
            mock.patch.object(server, "_run_executor_callback", side_effect=run_callback),
            mock.patch(
                "modiff.server.plan_hub_model_download",
                return_value=self._space_plan(revision=revision, snapshotCommit=revision),
            ),
            mock.patch(
                "modiff.server.download_hub_model",
                side_effect=fake_download,
            ) as download,
        ):
            await server._run_hf_download_task("unit/exact-model", entry)

        self.assertEqual(download.call_args.args[-1], revision)
        self.assertEqual(observed_reservations, [25])
        self.assertEqual(entry["reserved_bytes"], 0)

    async def test_app_refuses_download_when_queue_and_reserve_do_not_fit(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        server.hf_download_tasks["unit/already-queued"] = {"reserved_bytes": 500}
        revision = "a" * 40

        with (
            mock.patch(
                "modiff.server.plan_hub_model_download",
                return_value=self._space_plan(
                    revision=revision,
                    snapshotCommit=revision,
                    remainingBytes=400,
                    freeBytes=1000,
                    reserveBytes=200,
                ),
            ),
            mock.patch("modiff.server.download_hub_model") as download,
        ):
            response = await server.hf_download(
                FakeRequest(repo_id="unit/exact-model", revision=revision)
            )

        payload = json.loads(response.text)
        self.assertEqual(response.status, 507)
        self.assertEqual(payload["code"], "insufficient_model_download_space")
        self.assertFalse(payload["repair_required"])
        self.assertEqual(payload["result"]["downloadPlan"]["queuedReservationBytes"], 500)
        download.assert_not_called()

    async def test_app_refuses_download_when_immutable_size_is_unknown(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        revision = "a" * 40

        with (
            mock.patch(
                "modiff.server.plan_hub_model_download",
                return_value=self._space_plan(
                    revision=revision,
                    snapshotCommit=revision,
                    totalBytes=None,
                    remainingBytes=None,
                    sizeKnown=False,
                    fits=False,
                ),
            ),
            mock.patch("modiff.server.download_hub_model") as download,
        ):
            response = await server.hf_download(
                FakeRequest(repo_id="unit/exact-model", revision=revision)
            )

        payload = json.loads(response.text)
        self.assertEqual(response.status, 503)
        self.assertEqual(payload["code"], "huggingface_download_size_unknown")
        self.assertFalse(payload["repair_required"])
        download.assert_not_called()

    async def test_shared_memory_runtime_serializes_graph_and_download_model_io(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        server.serialize_model_io = True
        called = []

        await server.model_io_lock.acquire()
        pending = asyncio.create_task(
            server._run_executor_callback(lambda: called.append(True), serialize_model_io=True)
        )
        await asyncio.sleep(0.02)
        self.assertFalse(called)
        self.assertFalse(pending.done())

        server.model_io_lock.release()
        await pending
        self.assertTrue(called)

    async def test_discrete_runtime_keeps_model_io_concurrent(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        server.serialize_model_io = False
        called = []

        await server.model_io_lock.acquire()
        try:
            await server._run_executor_callback(lambda: called.append(True), serialize_model_io=True)
        finally:
            server.model_io_lock.release()
        self.assertTrue(called)

    async def test_concurrent_requests_join_one_app_download(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        release = asyncio.Event()
        calls = []

        async def fake_download(repo_id, entry):
            calls.append((repo_id, entry["task_id"]))
            await release.wait()
            return {"repo_id": repo_id, "complete": True, "repair_required": False}

        server._run_hf_download_task = fake_download
        first = asyncio.create_task(server.hf_download(FakeRequest(repo_id="unit/shared-model")))
        await asyncio.sleep(0)
        second = asyncio.create_task(server.hf_download(FakeRequest(repo_id="unit/shared-model")))
        await asyncio.sleep(0)

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(server.hf_download_tasks), 1)
        release.set()
        first_response, second_response = await asyncio.gather(first, second)
        first_payload = json.loads(first_response.text)
        second_payload = json.loads(second_response.text)
        self.assertFalse(first_payload["error"])
        self.assertEqual(first_payload["task_id"], second_payload["task_id"])
        self.assertNotIn("unit/shared-model", server.hf_download_tasks)

    async def test_incomplete_app_download_requires_repair_without_deleting_partial_state(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()

        async def fake_download(repo_id, entry):
            return {
                "repo_id": repo_id,
                "complete": False,
                "repair_required": True,
                "validation": {"reason": "One expected shard is incomplete."},
            }

        server._run_hf_download_task = fake_download
        response = await server.hf_download(FakeRequest(repo_id="unit/incomplete-model"))
        payload = json.loads(response.text)
        self.assertEqual(response.status, 409)
        self.assertTrue(payload["repair_required"])
        self.assertIn("incomplete", payload["error"])

    async def test_ltx_app_download_automatically_selects_only_diffusers_component_files(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        captured = {}

        async def fake_download(repo_id, entry):
            captured.update(entry)
            return {"repo_id": repo_id, "complete": True, "repair_required": False}

        server._run_hf_download_task = fake_download
        response = await server.hf_download(
            FakeRequest(repo_id="Lightricks/LTX-Video-0.9.8-13B-distilled")
        )
        payload = json.loads(response.text)

        self.assertFalse(payload["error"])
        self.assertEqual(len(captured["requested_files"]), 22)
        self.assertIn("model_index.json", captured["requested_files"])
        self.assertIn("transformer/diffusion_pytorch_model.safetensors.index.json", captured["requested_files"])
        self.assertIn("text_encoder/model-00004-of-00004.safetensors", captured["requested_files"])
        self.assertNotIn("ltxv-13b-0.9.8-dev.safetensors", captured["requested_files"])

    async def test_custom_download_carries_exact_commit_into_app_owned_task(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        captured = {}
        revision = "a" * 40

        async def fake_download(repo_id, entry):
            captured.update(entry)
            return {
                "repo_id": repo_id,
                "revision": entry["revision"],
                "complete": True,
                "repair_required": False,
            }

        server._run_hf_download_task = fake_download
        response = await server.hf_download(
            FakeRequest(repo_id="unit/exact-model", revision=revision)
        )
        payload = json.loads(response.text)

        self.assertFalse(payload["error"])
        self.assertEqual(captured["revision"], revision)
        self.assertEqual(payload["result"]["revision"], revision)

    async def test_cataloged_download_uses_reviewed_commit_when_client_omits_revision(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        captured = {}

        async def fake_download(repo_id, entry):
            captured.update(entry)
            return {
                "repo_id": repo_id,
                "revision": entry["revision"],
                "complete": True,
                "repair_required": False,
            }

        server._run_hf_download_task = fake_download
        response = await server.hf_download(
            FakeRequest(repo_id="Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers")
        )
        payload = json.loads(response.text)

        self.assertFalse(payload["error"])
        self.assertEqual(captured["revision"], "17c30769b1e0b5dcaa1799b117bf20a9c31f59d7")
        self.assertEqual(payload["result"]["revision"], captured["revision"])

    async def test_uncataloged_download_preserves_user_selected_revision_behavior(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        captured = {}

        async def fake_download(repo_id, entry):
            captured.update(entry)
            return {"repo_id": repo_id, "complete": True, "repair_required": False}

        server._run_hf_download_task = fake_download
        response = await server.hf_download(FakeRequest(repo_id="unit/custom-model"))

        self.assertFalse(json.loads(response.text)["error"])
        self.assertIsNone(captured["revision"])

    async def test_concurrent_download_rejects_a_different_exact_commit(self):
        server = WebServer(modules={})
        server.loop = asyncio.get_running_loop()
        release = asyncio.Event()

        async def fake_download(repo_id, entry):
            await release.wait()
            return {"repo_id": repo_id, "complete": True, "repair_required": False}

        server._run_hf_download_task = fake_download
        first = asyncio.create_task(
            server.hf_download(FakeRequest(repo_id="unit/exact-model", revision="a" * 40))
        )
        await asyncio.sleep(0)
        response = await server.hf_download(
            FakeRequest(repo_id="unit/exact-model", revision="b" * 40)
        )

        self.assertEqual(response.status, 409)
        self.assertIn("immutable snapshot", json.loads(response.text)["error"])
        release.set()
        await first


if __name__ == "__main__":
    unittest.main()
