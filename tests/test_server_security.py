import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from modiff.server import WebServer, is_cache_servable_data_type, is_hidden_path


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


class CacheRequest:
    def __init__(self, node, field):
        self.match_info = {"node": node, "field": field}
        self.query = {}
        self.headers = {}


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
        self.assertIn(("POST", "/template_gallery/install"), methods)
        self.assertIn(("GET", "/template_gallery/status"), methods)
        self.assertIn(("GET", "/template_gallery/plan"), methods)
        self.assertIn(("GET", "/hf_hub"), methods)
        self.assertIn(("GET", "/hf_cache"), methods)
        self.assertIn(("POST", "/hf_token"), methods)
        self.assertIn(("GET", "/model_artifact_catalog"), methods)
        self.assertNotIn(("GET", "/stop"), methods)
        self.assertNotIn(("GET", "/hf_download"), methods)
        self.assertNotIn(("GET", "/template_gallery/install"), methods)
        self.assertFalse(any(path.startswith("/inference/") for _method, path in methods))

    def test_hidden_path_supports_dotfiles_and_windows_attributes(self):
        dotfile = SimpleNamespace(name=".token", stat=lambda: (_ for _ in ()).throw(AssertionError("unused")))
        windows_hidden = SimpleNamespace(name="token", stat=lambda: SimpleNamespace(st_file_attributes=0x2))
        visible = SimpleNamespace(name="token", stat=lambda: SimpleNamespace(st_file_attributes=0))

        self.assertTrue(is_hidden_path(dotfile))
        with patch("modiff.server.stat.FILE_ATTRIBUTE_HIDDEN", 0x2, create=True):
            self.assertTrue(is_hidden_path(windows_hidden))
            self.assertFalse(is_hidden_path(visible))

    def test_cache_served_type_boundary_accepts_only_media_and_text_families(self):
        for value in (
            "image",
            "audio",
            "video",
            "text",
            "string",
            ["audio"],
            ("video",),
            {"text"},
        ):
            with self.subTest(value=value):
                self.assertTrue(is_cache_servable_data_type(value))
        for value in ("diffusers_auto_model", "any", "int", ["collection"], {}, None):
            with self.subTest(value=value):
                self.assertFalse(is_cache_servable_data_type(value))

    def test_mutation_origin_guard_is_registered_centrally(self):
        self.assertIn(self.server._mutation_origin_middleware, self.server.app.middlewares)

    async def test_cache_rejects_undeclared_dynamic_output_before_conversion(self):
        class OpaqueCacheValue:
            def __str__(self):
                raise AssertionError("Opaque cache output was converted to text.")

            def __bytes__(self):
                raise AssertionError("Opaque cache output was converted to bytes.")

            def __fspath__(self):
                raise AssertionError("Opaque cache output was converted to a path.")

            def __iter__(self):
                raise AssertionError("Opaque cache output was inspected as an iterable.")

            def __reduce__(self):
                raise AssertionError("Opaque cache output was serialized.")

        module = "modules.ModularDiffusers"
        action = "Denoise"
        self.server.node_cache["denoise"] = SimpleNamespace(
            module_name=module,
            class_name=action,
            output={"route_state_out": OpaqueCacheValue()},
            params={},
        )
        malformed_registries = {
            "invalid registry": [],
            "missing module": {},
            "invalid module": {module: []},
            "missing action": {module: {}},
            "invalid action": {module: {action: []}},
            "missing params": {module: {action: {}}},
            "invalid params": {module: {action: {"params": None}}},
            "invalid field definition": {
                module: {action: {"params": {"route_state_out": []}}}
            },
            "missing field type": {
                module: {action: {"params": {"route_state_out": {}}}}
            },
            "missing static field": {
                module: {action: {"params": {"image": {"type": "image"}}}}
            },
        }

        with (
            patch("modiff.server.to_bytes") as convert_media,
            patch("modiff.server.web.FileResponse") as file_response,
        ):
            for label, registry in malformed_registries.items():
                with self.subTest(label=label):
                    self.server.modules = registry
                    response = await self.server.cache(CacheRequest("denoise", "route_state_out"))
                    self.assertEqual(response.status, 400)
                    self.assertEqual(
                        response.text,
                        "Field route_state_out is not declared as a cache-served field for node denoise.",
                    )

        convert_media.assert_not_called()
        file_response.assert_not_called()

    async def test_cache_preserves_missing_field_404_and_declared_media_serving(self):
        module = "modules.StaticMedia"
        action = "Preview"
        cached_image = object()
        self.server.modules = {
            module: {
                action: {
                    "params": {
                        "image": {"type": "image", "fieldOptions": {"quality": 91}},
                    }
                }
            }
        }
        self.server.node_cache["preview"] = SimpleNamespace(
            module_name=module,
            class_name=action,
            output={"image": cached_image},
            params={},
        )

        missing = await self.server.cache(CacheRequest("preview", "absent"))
        self.assertEqual(missing.status, 404)
        self.assertEqual(missing.text, "Field absent not found in node preview cache.")

        with patch("modiff.server.to_bytes", return_value=b"encoded-image") as convert_media:
            response = await self.server.cache(CacheRequest("preview", "image"))

        self.assertEqual(response.status, 200)
        self.assertEqual(response.body, b"encoded-image")
        self.assertEqual(response.content_type, "image/webp")
        convert_media.assert_called_once_with(
            "image",
            cached_image,
            {"format": "WEBP", "quality": 100},
        )

    async def test_cache_rejects_statically_declared_connector_outputs_before_conversion(self):
        class OpaqueConnector:
            def __fspath__(self):
                raise AssertionError("Opaque connector output was converted to a path.")

            def __iter__(self):
                raise AssertionError("Opaque connector output was inspected as an iterable.")

        module = "modules.ModularDiffusers"
        action = "AutoModelLoader"
        self.server.modules = {
            module: {
                action: {
                    "params": {
                        "model": {
                            "display": "output",
                            "type": "diffusers_auto_model",
                        }
                    }
                }
            }
        }
        self.server.node_cache["loader"] = SimpleNamespace(
            module_name=module,
            class_name=action,
            output={"model": OpaqueConnector()},
            params={},
        )

        with (
            patch("modiff.server.to_bytes") as convert_media,
            patch("modiff.server.web.FileResponse") as file_response,
        ):
            response = await self.server.cache(CacheRequest("loader", "model"))

        self.assertEqual(response.status, 400)
        self.assertEqual(
            response.text,
            "Field model has a type that cannot be served from node cache.",
        )
        convert_media.assert_not_called()
        file_response.assert_not_called()

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

    async def test_parallel_model_discovery_refreshes_share_one_background_scan(self):
        actualizations = 0

        def actualize():
            nonlocal actualizations
            actualizations += 1

        with patch("modiff.server.modelstore.actualize", side_effect=actualize):
            await asyncio.gather(
                self.server._refresh_model_indexes(),
                self.server._refresh_model_indexes(),
                self.server._refresh_model_indexes(),
            )

        self.assertEqual(actualizations, 1)
        self.assertIsNone(self.server.model_discovery_task)

    async def test_parallel_cache_diagnostics_refreshes_coalesce_and_do_not_block_loop(self):
        actualizations = 0
        diagnostic_scans = 0
        scan_started = threading.Event()
        release_scan = threading.Event()

        def actualize():
            nonlocal actualizations
            actualizations += 1
            time.sleep(0.03)

        def diagnostics():
            nonlocal diagnostic_scans
            diagnostic_scans += 1
            scan_started.set()
            if not release_scan.wait(timeout=1):
                raise TimeoutError("The diagnostics test did not release its worker thread.")
            return {"locations": [{"label": "test cache", "repo_count": 1}]}

        requests = [JsonRequest({}) for _ in range(3)]
        for request in requests:
            request.query = {"refresh": "true"}

        with (
            patch("modiff.server.modelstore.actualize", side_effect=actualize),
            patch("modiff.server.get_cache_diagnostics", side_effect=diagnostics),
        ):
            tasks = [asyncio.create_task(self.server.model_cache_diagnostics(request)) for request in requests]
            started = await asyncio.wait_for(asyncio.to_thread(scan_started.wait, 0.5), timeout=0.75)
            self.assertTrue(started)
            heartbeat_started = time.monotonic()
            await asyncio.wait_for(asyncio.sleep(0.01), timeout=0.15)
            self.assertLess(time.monotonic() - heartbeat_started, 0.1)
            release_scan.set()
            responses = await asyncio.gather(*tasks)

            cached_request = JsonRequest({})
            cached_request.query = {"refresh": "false"}
            cached_response = await self.server.model_cache_diagnostics(cached_request)

        self.assertEqual(actualizations, 1)
        self.assertEqual(diagnostic_scans, 1)
        self.assertEqual([json.loads(response.text) for response in responses], [json.loads(cached_response.text)] * 3)
        self.assertIsNone(self.server.model_discovery_task)
        self.assertEqual(self.server.model_discovery_snapshot_tasks, {})

    async def test_hf_cache_validation_is_coalesced_off_loop_and_never_serves_stale_status_after_refresh(self):
        actualizations = 0
        inventory_scans = 0
        scan_started = threading.Event()
        release_scan = threading.Event()

        def actualize():
            nonlocal actualizations
            actualizations += 1
            time.sleep(0.03)

        def inventory():
            nonlocal inventory_scans
            inventory_scans += 1
            scan_started.set()
            if not release_scan.wait(timeout=1):
                raise TimeoutError("The inventory test did not release its worker thread.")
            return [
                {
                    "id": "owner/model",
                    "class_names": ["TestPipeline"],
                    "cached": True,
                    "installed": True,
                    "complete": True,
                    "repair_required": False,
                    "install_reason": "Complete test snapshot.",
                    "active_files": [],
                    "missing_files": [],
                    "corrupt_files": [],
                    "planned_revision": "a" * 40,
                    "planned_files": ["model.safetensors"],
                }
            ]

        requests = [JsonRequest({}) for _ in range(2)]
        for request in requests:
            request.query = {"refresh": "true"}

        with (
            patch("modiff.server.modelstore.actualize", side_effect=actualize),
            patch.object(self.server, "_build_hf_cache_inventory", side_effect=inventory) as build_inventory,
        ):
            tasks = [asyncio.create_task(self.server.hf_cache(request)) for request in requests]
            started = await asyncio.wait_for(asyncio.to_thread(scan_started.wait, 0.5), timeout=0.75)
            self.assertTrue(started)
            heartbeat_started = time.monotonic()
            await asyncio.wait_for(asyncio.sleep(0.01), timeout=0.15)
            self.assertLess(time.monotonic() - heartbeat_started, 0.1)
            release_scan.set()
            responses = await asyncio.gather(*tasks)

            self.assertEqual(actualizations, 1)
            self.assertEqual(inventory_scans, 1)
            self.assertTrue(all(json.loads(response.text)[0]["installed"] for response in responses))

            # A new refresh advances the generation. Failure to validate that
            # generation must propagate; the older installed=True payload is
            # never used as a fallback.
            build_inventory.side_effect = RuntimeError("fresh validation failed")
            failing_request = JsonRequest({})
            failing_request.query = {"refresh": "true"}
            with self.assertRaisesRegex(RuntimeError, "fresh validation failed"):
                await self.server.hf_cache(failing_request)

        self.assertEqual(actualizations, 2)
        self.assertEqual(self.server.model_discovery_generation, 2)
        self.assertEqual(self.server.model_discovery_snapshot_tasks, {})

    async def test_local_model_post_refresh_filtering_does_not_block_loop(self):
        filter_started = threading.Event()
        release_filter = threading.Event()

        def local_ids(_match):
            filter_started.set()
            if not release_filter.wait(timeout=1):
                raise TimeoutError("The local-model test did not release its worker thread.")
            return ["weights/example.safetensors"]

        request = JsonRequest({})
        request.query = {"refresh": "true", "match": "example"}
        with (
            patch("modiff.server.modelstore.actualize"),
            patch("modiff.server.modelstore.get_local_ids", side_effect=local_ids),
        ):
            task = asyncio.create_task(self.server.local_models(request))
            started = await asyncio.wait_for(asyncio.to_thread(filter_started.wait, 0.5), timeout=0.75)
            self.assertTrue(started)
            heartbeat_started = time.monotonic()
            await asyncio.wait_for(asyncio.sleep(0.01), timeout=0.15)
            self.assertLess(time.monotonic() - heartbeat_started, 0.1)
            release_filter.set()
            response = await task

        self.assertEqual(json.loads(response.text), ["weights/example.safetensors"])

    async def test_slow_model_capabilities_build_does_not_block_health_and_coalesces(self):
        capability_builds = 0
        build_started = threading.Event()
        release_build = threading.Event()

        def build_capabilities(query):
            nonlocal capability_builds
            capability_builds += 1
            build_started.set()
            if not release_build.wait(timeout=1):
                raise TimeoutError("The capability test did not release its worker thread.")
            return {"error": False, "count": 0, "capabilities": [], "query": query}

        capability_requests = [JsonRequest({}) for _ in range(2)]
        for request in capability_requests:
            request.query = {"q": "QWEN"}

        with (
            patch.object(
                self.server,
                "_build_model_capabilities_payload",
                side_effect=build_capabilities,
            ),
            patch.object(
                self.server,
                "_build_runtime_status_payload",
                return_value={"error": False, "ready": True},
            ),
        ):
            capability_tasks = [
                asyncio.create_task(self.server.model_capabilities(request))
                for request in capability_requests
            ]
            started = await asyncio.wait_for(asyncio.to_thread(build_started.wait, 0.5), timeout=0.75)
            self.assertTrue(started)

            health_started = time.monotonic()
            health_response = await asyncio.wait_for(
                self.server.runtime_status(JsonRequest({})),
                timeout=0.15,
            )
            self.assertLess(time.monotonic() - health_started, 0.1)
            self.assertTrue(json.loads(health_response.text)["ready"])

            release_build.set()
            capability_responses = await asyncio.gather(*capability_tasks)

        self.assertEqual(capability_builds, 1)
        self.assertEqual(
            [json.loads(response.text)["query"] for response in capability_responses],
            ["qwen", "qwen"],
        )
        self.assertEqual(self.server.control_snapshot_tasks, {})

    async def test_cancelled_resource_poll_does_not_cancel_other_pollers(self):
        release = threading.Event()
        started = threading.Event()

        def build_resources():
            started.set()
            if not release.wait(5):
                raise TimeoutError("The resource test did not release its worker.")
            return {"schemaVersion": 1, "sampledAt": 123}

        with patch.object(self.server, "_runtime_resource_snapshot", side_effect=build_resources) as build:
            first = asyncio.create_task(self.server.runtime_resources(JsonRequest({})))
            second = asyncio.create_task(self.server.runtime_resources(JsonRequest({})))
            try:
                async with asyncio.timeout(1):
                    while not started.is_set():
                        await asyncio.sleep(0.01)
                first.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                self.assertFalse(second.done())
            finally:
                release.set()
                responses = await asyncio.gather(first, second, return_exceptions=True)
            self.assertEqual(json.loads(responses[1].text), {"schemaVersion": 1, "sampledAt": 123})
            build.assert_called_once_with()
        self.assertEqual(self.server.control_snapshot_tasks, {})

    async def test_duplicate_resource_polls_cannot_fill_workers_and_starve_health(self):
        # A native/system probe can be slow even when it releases the GIL.
        # Duplicate waiters must not each occupy another default-pool worker.
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=2))
        release = threading.Event()
        builds = []

        def build_resources():
            builds.append(threading.get_ident())
            if not release.wait(5):
                raise TimeoutError("The resource test did not release its worker.")
            return {"schemaVersion": 1, "accelerators": [], "currentRun": None}

        with (
            patch.object(self.server, "_runtime_resource_snapshot", side_effect=build_resources),
            patch.object(self.server, "_build_runtime_status_payload", return_value={"ready": True}),
        ):
            polls = [asyncio.create_task(self.server.runtime_resources(JsonRequest({}))) for _ in range(4)]
            try:
                await asyncio.sleep(0.05)
                health = await asyncio.wait_for(self.server.runtime_status(JsonRequest({})), timeout=0.5)
                self.assertTrue(json.loads(health.text)["ready"])
                self.assertEqual(len(builds), 1)
            finally:
                release.set()
                results = await asyncio.gather(*polls, return_exceptions=True)
        self.assertTrue(all(not isinstance(result, BaseException) for result in results))
        self.assertEqual([json.loads(result.text) for result in results], [
            {"schemaVersion": 1, "accelerators": [], "currentRun": None}
        ] * 4)
        self.assertEqual(self.server.control_snapshot_tasks, {})

    async def test_slow_health_probe_is_coalesced_off_loop(self):
        health_builds = 0
        build_started = threading.Event()
        release_build = threading.Event()

        def build_health():
            nonlocal health_builds
            health_builds += 1
            build_started.set()
            if not release_build.wait(timeout=1):
                raise TimeoutError("The health test did not release its worker thread.")
            return {"error": False, "ready": True}

        with patch.object(self.server, "_build_runtime_status_payload", side_effect=build_health):
            health_tasks = [
                asyncio.create_task(self.server.runtime_status(JsonRequest({})))
                for _ in range(3)
            ]
            started = await asyncio.wait_for(asyncio.to_thread(build_started.wait, 0.5), timeout=0.75)
            self.assertTrue(started)
            heartbeat_started = time.monotonic()
            await asyncio.wait_for(asyncio.sleep(0.01), timeout=0.15)
            self.assertLess(time.monotonic() - heartbeat_started, 0.1)
            release_build.set()
            responses = await asyncio.gather(*health_tasks)

        self.assertEqual(health_builds, 1)
        self.assertTrue(all(json.loads(response.text)["ready"] for response in responses))
        self.assertEqual(self.server.control_snapshot_tasks, {})

    async def test_primed_multi_megabyte_capability_response_never_serializes_during_health(self):
        # Exercise the real serializer and a payload at the same scale as the
        # production catalog. Priming happens before the listener is opened;
        # request handlers must then reuse the immutable bytes directly.
        large_payload = {
            "error": False,
            "schemaVersion": 2,
            "capabilities": [],
            "serializedContractFixture": "x" * (8 * 1024 * 1024),
        }
        with patch.object(
            self.server,
            "_build_model_capabilities_payload",
            return_value=large_payload,
        ) as build_capabilities:
            primed = await self.server._model_capabilities_response("")

        self.assertGreater(len(primed), 8 * 1024 * 1024)
        build_capabilities.assert_called_once_with("")

        request = JsonRequest({})
        request.query = {}
        with (
            patch.object(
                self.server,
                "_build_model_capabilities_payload",
                side_effect=AssertionError("A primed request must not rebuild or reserialize the catalog."),
            ),
            patch.object(
                self.server,
                "_build_runtime_status_payload",
                return_value={"error": False, "ready": True},
            ),
        ):
            capability_tasks = [
                asyncio.create_task(self.server.model_capabilities(request))
                for _ in range(2)
            ]
            health_started = time.monotonic()
            health_response = await asyncio.wait_for(
                self.server.runtime_status(JsonRequest({})),
                timeout=0.15,
            )
            self.assertLess(time.monotonic() - health_started, 0.1)
            capability_responses = await asyncio.gather(*capability_tasks)

        self.assertTrue(json.loads(health_response.text)["ready"])
        self.assertEqual([len(response.body) for response in capability_responses], [len(primed), len(primed)])
        self.assertEqual(self.server.control_snapshot_tasks, {})

    async def test_false_model_discovery_query_does_not_refresh(self):
        request = JsonRequest({})
        request.query = {"refresh": "false", "compact": "1"}
        refresh = AsyncMock()

        with (
            patch.object(self.server, "_refresh_model_indexes", refresh),
            patch("modiff.server.modelstore.get_hf_models", return_value=[]),
        ):
            response = await self.server.hf_cache(request)

        self.assertEqual(response.status, 200)
        refresh.assert_not_awaited()

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
        self.server.hf_download_tasks = {
            "unit/model": {
                "task_id": "download-task",
                "started_at": 1.0,
                "revision": "a" * 40,
                "requested_files": ["weights/model.safetensors"],
                "reserved_bytes": 25,
                "progress": {
                    "type": "hf_download_progress",
                    "repo_id": "unit/model",
                    "task_id": "download-task",
                    "download_id": "download-task",
                    "status": "downloading",
                    "phase": "downloading",
                    "progress": 0.75,
                    "remaining_bytes": 25,
                    "started_at": 1.0,
                    "updated_at": 2.0,
                },
            }
        }
        websocket = EmptyWebSocket()

        with patch("modiff.server.web.WebSocketResponse", return_value=websocket):
            response = await self.server.websocket(WebSocketRequest(origin="http://localhost:5173"))

        self.assertIs(response, websocket)
        self.assertTrue(websocket.prepared)
        self.assertEqual(websocket.messages[0]["type"], "welcome")
        self.assertNotIn("workflow_snapshot", websocket.messages[0]["recent"][0])
        self.assertTrue(websocket.messages[0]["recent"][0]["has_workflow_snapshot"])
        self.assertEqual(websocket.messages[0]["downloads"][0]["repo_id"], "unit/model")
        self.assertNotIn("requested_files", websocket.messages[0]["downloads"][0])

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

    async def test_signal_lookup_dispatches_directly_without_background_broadcast_backlog(self):
        self.server.loop = asyncio.get_running_loop()
        self.server.ws_sessions["signal-session"] = EmptyWebSocket()
        broadcasts = []

        async def direct_broadcast(message, sid):
            broadcasts.append((message, sid))
            self.server.pending_ws_requests[message["request_id"]].set_result("StableDiffusionXLModularPipeline")

        def reject_queued_dispatch(*_args, **_kwargs):
            raise AssertionError("Synchronous signal lookup must not use the background broadcast queue.")

        self.server.broadcast = direct_broadcast
        self.server.queue_message = reject_queued_dispatch

        result = await asyncio.to_thread(
            self.server.get_signal_value,
            "guider-node",
            "guider_out",
            "signal-session",
            1,
        )

        self.assertEqual(result, "StableDiffusionXLModularPipeline")
        self.assertEqual(len(broadcasts), 1)
        self.assertEqual(broadcasts[0][0]["type"], "get_signal_value")
        self.assertEqual(broadcasts[0][1], "signal-session")
        self.assertEqual(self.server.pending_ws_requests, {})

    async def test_signal_lookup_timeout_cleans_loop_owned_pending_request(self):
        self.server.loop = asyncio.get_running_loop()
        self.server.ws_sessions["signal-session"] = EmptyWebSocket()

        async def unanswered_broadcast(_message, _sid):
            return None

        self.server.broadcast = unanswered_broadcast
        result = await asyncio.to_thread(
            self.server.get_signal_value,
            "guider-node",
            "guider_out",
            "signal-session",
            0.01,
        )
        await asyncio.sleep(0)

        self.assertEqual(result, {"__MODIFF_ERROR": "timeout"})
        self.assertEqual(self.server.pending_ws_requests, {})

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
