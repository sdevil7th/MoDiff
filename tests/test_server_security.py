import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from modiff.server import WebServer, is_hidden_path


class JsonRequest:
    def __init__(self, payload, *, origin=None, host="127.0.0.1:8088"):
        self._payload = payload
        self.headers = {"Origin": origin} if origin else {}
        self.host = host
        self.query = {}
        self.match_info = {}
        self.can_read_body = True

    async def json(self):
        return self._payload


class WebSocketRequest:
    def __init__(self, *, origin=None, host="127.0.0.1:8088", remote="127.0.0.1", sid="test-session"):
        self.headers = {"Origin": origin} if origin else {}
        self.host = host
        self.remote = remote
        self.query = {"sid": sid}


class EmptyWebSocket:
    def __init__(self):
        self.closed = False
        self.prepared = False
        self.messages = []

    async def prepare(self, _request):
        self.prepared = True

    async def send_json(self, message):
        self.messages.append(message)

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class ServerSecurityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.server = WebServer(
            modules={},
            work_dir=self.temporary.name,
            data_dir=self.temporary.name,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_state_changing_control_routes_are_post_only(self):
        methods = {(route.method, route.resource.canonical) for route in self.server.app.router.routes()}
        self.assertIn(("POST", "/stop"), methods)
        self.assertIn(("POST", "/hf_download"), methods)
        self.assertIn(("GET", "/hf_hub"), methods)
        self.assertIn(("GET", "/hf_cache"), methods)
        self.assertIn(("POST", "/hf_token"), methods)
        self.assertIn(("GET", "/model_artifact_catalog"), methods)
        self.assertNotIn(("GET", "/stop"), methods)
        self.assertNotIn(("GET", "/hf_download"), methods)
        self.assertFalse(any(path.startswith("/inference/") for _method, path in methods))

    def test_hidden_path_supports_dotfiles_and_windows_attributes(self):
        dotfile = SimpleNamespace(name=".token", stat=lambda: (_ for _ in ()).throw(AssertionError("unused")))
        windows_hidden = SimpleNamespace(name="token", stat=lambda: SimpleNamespace(st_file_attributes=0x2))
        visible = SimpleNamespace(name="token", stat=lambda: SimpleNamespace(st_file_attributes=0))

        self.assertTrue(is_hidden_path(dotfile))
        with patch("modiff.server.stat.FILE_ATTRIBUTE_HIDDEN", 0x2, create=True):
            self.assertTrue(is_hidden_path(windows_hidden))
            self.assertFalse(is_hidden_path(visible))

    def test_mutation_origin_guard_is_registered_centrally(self):
        self.assertIn(self.server._mutation_origin_middleware, self.server.app.middlewares)

    def test_template_gallery_route_is_optional_for_remote_asset_builds(self):
        missing_gallery = Path(self.temporary.name) / "no-local-gallery"
        with patch("modiff.server.TEMPLATE_GALLERY_ROOT", missing_gallery):
            remote_server = WebServer(
                modules={},
                work_dir=self.temporary.name,
                data_dir=self.temporary.name,
            )

        routes = {route.resource.canonical for route in remote_server.app.router.routes()}
        self.assertNotIn("/template-gallery", routes)
        self.assertIn("/assets", routes)

    def test_template_gallery_route_remains_available_for_offline_builds(self):
        local_gallery = Path(self.temporary.name) / "template-gallery"
        local_gallery.mkdir()
        with patch("modiff.server.TEMPLATE_GALLERY_ROOT", local_gallery):
            offline_server = WebServer(
                modules={},
                work_dir=self.temporary.name,
                data_dir=self.temporary.name,
            )

        routes = {route.resource.canonical for route in offline_server.app.router.routes()}
        self.assertIn("/template-gallery", routes)

    async def test_all_methods_reject_hostile_dns_host_and_origin(self):
        called = False

        async def handler(_request):
            nonlocal called
            called = True
            return web.json_response({"ok": True})

        for method in ("GET", "POST", "PUT", "PATCH", "DELETE"):
            with self.subTest(method=method):
                response = await self.server._mutation_origin_middleware(
                    SimpleNamespace(
                        method=method,
                        headers={"Origin": "https://attacker.example"},
                        host="attacker.example",
                        remote="127.0.0.1",
                    ),
                    handler,
                )
                self.assertEqual(response.status, 403)
                self.assertEqual(json.loads(response.text)["code"], "untrusted_request_boundary")
        self.assertFalse(called)

    async def test_read_rejects_hostile_origin_even_with_loopback_destination(self):
        called = False

        async def handler(_request):
            nonlocal called
            called = True
            return web.json_response({"ok": True})

        response = await self.server._mutation_origin_middleware(
            SimpleNamespace(
                method="GET",
                headers={"Origin": "https://attacker.example"},
                host="127.0.0.1:8088",
                remote="127.0.0.1",
            ),
            handler,
        )

        self.assertEqual(response.status, 403)
        self.assertEqual(json.loads(response.text)["code"], "untrusted_request_boundary")
        self.assertFalse(called)

    async def test_loopback_vite_origin_can_mutate_loopback_backend(self):
        async def handler(_request):
            return web.json_response({"ok": True})

        response = await self.server._mutation_origin_middleware(
            SimpleNamespace(
                method="POST",
                headers={"Origin": "http://localhost:5173"},
                host="127.0.0.1:8088",
                remote="127.0.0.1",
            ),
            handler,
        )

        self.assertEqual(response.status, 200)
        self.assertTrue(json.loads(response.text)["ok"])

    async def test_originless_native_mutation_requires_loopback_host_and_peer(self):
        async def handler(_request):
            return web.json_response({"ok": True})

        local = await self.server._mutation_origin_middleware(
            SimpleNamespace(
                method="DELETE",
                headers={},
                host="localhost:8088",
                remote="::1",
            ),
            handler,
        )
        remote = await self.server._mutation_origin_middleware(
            SimpleNamespace(
                method="DELETE",
                headers={},
                host="127.0.0.1:8088",
                remote="192.0.2.10",
            ),
            handler,
        )

        self.assertEqual(local.status, 200)
        self.assertEqual(remote.status, 403)

    async def test_hostile_origin_is_rejected_even_with_loopback_request_host(self):
        async def handler(_request):
            return web.json_response({"ok": True})

        response = await self.server._mutation_origin_middleware(
            SimpleNamespace(
                method="POST",
                headers={"Origin": "https://attacker.example"},
                host="127.0.0.1:8088",
                remote="127.0.0.1",
            ),
            handler,
        )

        self.assertEqual(response.status, 403)

    async def test_registered_middleware_guards_real_read_and_mutation_routes(self):
        client = TestClient(TestServer(self.server.app))
        await client.start_server()
        try:
            hostile = await client.post(
                "/stop",
                headers={
                    "Host": "attacker.example",
                    "Origin": "https://attacker.example",
                },
            )
            hostile_payload = await hostile.json()
            hostile_read = await client.get(
                "/queue",
                headers={"Host": "attacker.example"},
            )
            hostile_read_payload = await hostile_read.json()
            local = await client.post(
                "/stop",
                headers={"Origin": "http://localhost:5173"},
            )
        finally:
            await client.close()

        self.assertEqual(hostile.status, 403)
        self.assertEqual(hostile_payload["code"], "untrusted_request_boundary")
        self.assertEqual(hostile_read.status, 403)
        self.assertEqual(hostile_read_payload["code"], "untrusted_request_boundary")
        self.assertEqual(local.status, 200)

    async def test_websocket_rejects_hostile_browser_origin_before_upgrade(self):
        with patch("modiff.server.web.WebSocketResponse") as websocket_factory:
            response = await self.server.websocket(WebSocketRequest(origin="https://attacker.example"))

        self.assertEqual(response.status, 403)
        websocket_factory.assert_not_called()

    async def test_websocket_allows_loopback_origin_and_compacts_welcome_history(self):
        self.server.recent_tasks = [
            {
                "task_id": "completed-task",
                "status": "completed",
                "workflow_snapshot": {"nodes": [{"prompt": "private prompt"}]},
            }
        ]
        websocket = EmptyWebSocket()

        with patch("modiff.server.web.WebSocketResponse", return_value=websocket):
            response = await self.server.websocket(WebSocketRequest(origin="http://localhost:5173"))

        self.assertIs(response, websocket)
        self.assertTrue(websocket.prepared)
        self.assertEqual(websocket.messages[0]["type"], "welcome")
        self.assertNotIn("workflow_snapshot", websocket.messages[0]["recent"][0])
        self.assertTrue(websocket.messages[0]["recent"][0]["has_workflow_snapshot"])

    async def test_websocket_without_origin_requires_a_loopback_native_client(self):
        local_websocket = EmptyWebSocket()
        with patch("modiff.server.web.WebSocketResponse", return_value=local_websocket):
            local_response = await self.server.websocket(WebSocketRequest(origin=None))

        self.assertIs(local_response, local_websocket)
        self.assertTrue(local_websocket.prepared)

        with patch("modiff.server.web.WebSocketResponse") as websocket_factory:
            remote_response = await self.server.websocket(WebSocketRequest(origin=None, remote="192.0.2.10"))

        self.assertEqual(remote_response.status, 403)
        websocket_factory.assert_not_called()

    async def test_public_workflow_share_does_not_expose_backend_paths(self):
        encoded = base64.b64encode(b"small-preview").decode("ascii")
        response = await self.server.workflow_share_post(
            JsonRequest(
                {
                    "share_id": "safe-share",
                    "metadata": {"preview": f"data:image/png;base64,{encoded}"},
                    "manifest": {"media": {}},
                    "latestOutput": {},
                }
            )
        )
        payload = json.loads(response.text)
        serialized = json.dumps(payload)

        self.assertEqual(response.status, 200)
        self.assertNotIn(self.temporary.name, serialized)
        self.assertNotIn("backendShareMediaPath", serialized)
        self.assertNotIn('"path"', serialized)
        self.assertEqual(payload["persistedMedia"]["filename"], "preview.png")

        share_path = Path(self.temporary.name) / "studio" / "shares" / "safe-share.json"
        self.assertNotIn(self.temporary.name, share_path.read_text(encoding="utf-8"))

    def test_legacy_workflow_share_paths_are_sanitized_on_read(self):
        public = self.server._public_workflow_share(
            {
                "persistedMedia": {"path": "/private/share.png", "url": "/media/share.png"},
                "package": {
                    "manifest": {"media": {"backendShareMediaPath": "/private/share.png"}},
                    "latestOutput": {"backendShareMediaPath": "/private/share.png"},
                },
            }
        )
        self.assertNotIn("/private", json.dumps(public))


if __name__ == "__main__":
    unittest.main()
