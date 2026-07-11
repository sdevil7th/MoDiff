import asyncio
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


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

from modiff.server import WebServer  # noqa: E402


class FakeRequest:
    def __init__(self, payload=None, match_info=None, query=None):
        self._payload = payload
        self.match_info = match_info or {}
        self.query = query or {}

    async def json(self):
        return self._payload


def response_json(response):
    return json.loads(response.text)


class StudioBlockPersistenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    async def make_server(self):
        server = WebServer(work_dir=self.temp_dir.name, data_dir=self.temp_dir.name)
        await asyncio.sleep(0)
        return server

    async def test_block_save_list_get_and_delete(self):
        server = await self.make_server()
        payload = {
            "id": "unit-block",
            "name": "Unit Block",
            "version": 1,
            "nodes": [{"id": "node-1", "data": {"params": {}}, "position": {"x": 0, "y": 0}}],
            "edges": [],
            "inputs": [],
            "outputs": [],
            "exposedParams": [],
            "createdAt": 1,
            "updatedAt": 2,
        }

        post_response = await server.studio_blocks_post(FakeRequest(payload))
        self.assertEqual(post_response.status, 200)
        self.assertEqual(response_json(post_response)["block"]["id"], "unit-block")

        list_response = await server.studio_blocks_get(FakeRequest(query={}))
        list_payload = response_json(list_response)
        self.assertEqual(list_payload["count"], 1)
        self.assertEqual(list_payload["blocks"][0]["name"], "Unit Block")

        get_response = await server.studio_block_get(FakeRequest(match_info={"block_id": "unit-block"}))
        self.assertEqual(response_json(get_response)["block"]["id"], "unit-block")

        delete_response = await server.studio_block_delete(FakeRequest(match_info={"block_id": "unit-block"}))
        self.assertEqual(response_json(delete_response)["id"], "unit-block")

        empty_response = await server.studio_blocks_get(FakeRequest(query={}))
        self.assertEqual(response_json(empty_response)["count"], 0)

    async def test_invalid_block_is_rejected(self):
        server = await self.make_server()
        response = await server.studio_blocks_post(FakeRequest({"id": "broken", "name": "Broken"}))

        self.assertEqual(response.status, 400)
        self.assertTrue(response_json(response)["error"])


if __name__ == "__main__":
    unittest.main()
