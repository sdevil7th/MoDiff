import asyncio
import json
import unittest

from modiff.server import WebServer


class FakeRequest:
    def __init__(self, **query):
        self.query = query


class HuggingFaceDownloadConcurrencyTests(unittest.IsolatedAsyncioTestCase):
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


if __name__ == "__main__":
    unittest.main()
