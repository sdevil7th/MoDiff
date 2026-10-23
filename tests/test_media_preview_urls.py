import hashlib
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.io import wavfile
from unittest.mock import patch
from urllib.parse import unquote, urlsplit

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from modiff import media_io, server as server_module
from modiff.media_io import export_media_file
from modiff.NodeBase import NodeBase
from modiff.server import (
    WebServer,
    audio_download_filename,
    byte_range_response,
    file_backed_media_preview,
    parse_audio_download_sample_rate,
    resample_wav_bytes,
)


class MediaPreviewUrlTests(unittest.TestCase):
    def test_cache_url_preserves_ascii_and_encodes_each_identity_once(self):
        self.assertEqual(server_module.cache_media_url("preview", "images", 0), "/cache/preview/images/0")
        node, field = "visual-output/stage-preview %2F?#", "images/output %?#"
        url = server_module.cache_media_url(node, field, 2)
        parts = urlsplit(url).path.split("/")
        self.assertEqual(len(parts), 5)
        self.assertEqual([unquote(parts[2]), unquote(parts[3]), parts[4]], [node, field, "2"])
        self.assertEqual(urlsplit(url).query, "")
        self.assertEqual(urlsplit(url).fragment, "")

    def test_file_backed_media_uses_file_route(self):
        url = file_backed_media_preview("audio/source track.wav")

        self.assertTrue(url.startswith("/file?file=audio%2Fsource%20track.wav&t="))

    def test_existing_browser_url_is_preserved(self):
        self.assertEqual(
            file_backed_media_preview("https://example.test/source.wav"),
            "https://example.test/source.wav",
        )

    def test_non_path_media_uses_cache_fallback(self):
        self.assertIsNone(file_backed_media_preview({"samples": []}))

    def test_cached_media_supports_browser_byte_ranges(self):
        response = byte_range_response(
            SimpleNamespace(headers={"Range": "bytes=2-5"}),
            b"0123456789",
            content_type="audio/wav",
            filename="output.wav",
        )

        self.assertEqual(response.status, 206)
        self.assertEqual(response.body, b"2345")
        self.assertEqual(response.headers["Accept-Ranges"], "bytes")
        self.assertEqual(response.headers["Content-Range"], "bytes 2-5/10")
        self.assertEqual(response.headers["Content-Length"], "4")

    def test_cached_media_rejects_unsatisfiable_range(self):
        response = byte_range_response(
            SimpleNamespace(headers={"Range": "bytes=20-30"}),
            b"0123456789",
            content_type="audio/wav",
        )

        self.assertEqual(response.status, 416)
        self.assertEqual(response.headers["Content-Range"], "bytes */10")

    def test_download_sample_rate_rewrites_the_wav_header_and_duration(self):
        source = BytesIO()
        wavfile.write(source, 48000, np.arange(48000, dtype=np.int16))

        converted = resample_wav_bytes(source.getvalue(), 44100)
        converted_rate, converted_samples = wavfile.read(BytesIO(converted))

        self.assertEqual(converted_rate, 44100)
        self.assertEqual(converted_samples.shape[0], 44100)

    def test_download_sample_rate_contract_rejects_unsupported_rates(self):
        self.assertEqual(parse_audio_download_sample_rate("44100"), 44100)
        self.assertIsNone(parse_audio_download_sample_rate(None))
        with self.assertRaisesRegex(ValueError, "must be one of"):
            parse_audio_download_sample_rate("22050")

    def test_download_filename_records_the_actual_export_rate(self):
        self.assertEqual(audio_download_filename("mix.wav", 44100), "mix-44.1kHz.wav")

    def test_concurrent_media_exports_create_one_atomic_cached_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (4, 3), "navy").save(source)
            export_root = root / "exports"

            def fake_export(_source, destination, _format_id, _options):
                destination.write_bytes(b"encoded")

            with patch.object(media_io, "_image_export", side_effect=fake_export) as encode:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(
                        executor.map(
                            lambda _index: export_media_file(
                                source,
                                kind="image",
                                format_id="png",
                                cache_root=export_root,
                            ),
                            range(4),
                        )
                    )
            leftovers = list(export_root.glob(".*"))

        self.assertEqual(len({result[0] for result in results}), 1)
        self.assertEqual(encode.call_count, 1)
        self.assertEqual(leftovers, [])


class GeneratedCacheUrlTests(unittest.IsolatedAsyncioTestCase):
    def dispatch(self, directory, node_id, field, value, data_type, display, *, task=False):
        class Preview(NodeBase):
            def execute(self):
                return {field: [value]}

        module = ".".join(Preview.__module__.split(".")[:-1])
        definition = {module: {"Preview": {"params": {
            field: {"type": data_type, "display": "output"},
            "preview": {"type": "url", "display": display, "fieldOptions": {"format": "PNG"}},
        }}}}
        server = WebServer(modules=definition, work_dir=directory, data_dir=directory)
        messages = []
        server.queue_message = lambda message, *_args, **_kwargs: messages.append(message)
        if task:
            server.current_task = {"task_id": "cache-url-run", "attempt_index": 0}
        with patch("modiff.NodeBase._module_map", return_value=definition), patch("modiff.NodeBase._server", return_value=server):
            server.node_cache[node_id] = Preview(node_id)
            server.execute_node(node_id, {"module": module, "action": "Preview", "params": {
                "preview": {"display": display, "sourceKey": field},
            }}, "test", quiet=True)
        message = next(message for message in messages if message.get("type") == "update_value")
        return server, message

    async def test_all_cache_url_branches_preserve_reserved_node_and_field_characters(self):
        cases = [("image", "ui_image", Image.new("RGB", (2, 3), "navy")),
                 ("audio", "ui_audio", {"samples": [], "sample_rate": 8000}),
                 ("string", "ui_text", "text")]
        with tempfile.TemporaryDirectory() as directory:
            for data_type, display, value in cases:
                with self.subTest(display=display):
                    node, field = "visual-output/stage-preview %?#", "output/field %?#"
                    _server, message = self.dispatch(directory, node, field, value, data_type, display)
                    url = message["value"][0]
                    parts = urlsplit(url).path.split("/")
                    self.assertEqual(len(parts), 5)
                    self.assertEqual([unquote(parts[2]), unquote(parts[3])], [node, field])

    async def test_slash_id_image_serves_and_persists_after_cache_is_cleared(self):
        with tempfile.TemporaryDirectory() as directory:
            server, message = self.dispatch(directory, "visual-output/stage-preview", "images/output %?#",
                                            Image.new("RGB", (2, 3), "navy"), "image", "ui_image", task=True)
            url = message["value"][0]
            self.assertIn("visual-output%2Fstage-preview", url)
            self.assertTrue(message["backend_persisted"])
            output = next(row for row in server._read_studio_output_state()["outputs"] if row["id"] == message["output_id"])
            item = output["mediaItems"][0]
            self.assertIn("backendPath", item)
            app = web.Application()
            app.router.add_get("/cache/{node}/{field}/{index}", server.cache)
            app.router.add_get("/file", server.fileGet)
            async with TestClient(TestServer(app)) as client:
                async with client.get(url) as response:
                    self.assertEqual(response.status, 200)
                    body = await response.read()
                self.assertEqual(Image.open(BytesIO(body)).size, (2, 3))
                self.assertEqual("sha256:bytes:" + hashlib.sha256(body).hexdigest(), item["mediaHash"])
                server.node_cache.clear()
                async with client.get(item["url"]) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(await response.read(), body)


class WorkspaceFileRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "work"
        self.workspace.mkdir()
        self.sibling = self.root / "work-secret"
        self.sibling.mkdir()
        self.secret = self.sibling / "secret.png"
        Image.new("RGB", (2, 2), "red").save(self.secret)
        self.server = WebServer(modules={}, work_dir=str(self.workspace), data_dir=str(self.workspace))

    def tearDown(self):
        self.temporary.cleanup()

    async def test_file_routes_reject_traversal_and_sibling_prefixes(self):
        list_response = await self.server.listdir(
            SimpleNamespace(query={"path": "../work-secret"})
        )
        preview_response = await self.server.preview(
            SimpleNamespace(query={"file": str(self.secret)})
        )
        stream_response = await self.server.stream(
            SimpleNamespace(query={"file": "../work-secret/secret.png"})
        )

        self.assertEqual(list_response.status, 403)
        self.assertEqual(preview_response.status, 403)
        self.assertEqual(stream_response.status, 403)
        self.assertIn("outside", json.loads(preview_response.text)["error"])

    async def test_file_routes_reject_resolved_link_escapes(self):
        original_resolve = Path.resolve

        def resolve_with_escape(path, strict=False):
            if path.name == "secret.png" and path.parent.name == "linked-secret":
                return self.secret
            return original_resolve(path, strict=strict)

        with patch.object(Path, "resolve", resolve_with_escape):
            response = await self.server.stream(
                SimpleNamespace(query={"file": "linked-secret/secret.png"})
            )

        self.assertEqual(response.status, 403)

    async def test_valid_workspace_files_remain_available(self):
        image_path = self.workspace / "image.png"
        Image.new("RGB", (3, 2), "blue").save(image_path)

        listing = json.loads((await self.server.listdir(SimpleNamespace(query={"path": "."}))).text)
        preview = await self.server.preview(SimpleNamespace(query={"file": "image.png"}))
        stream = await self.server.stream(SimpleNamespace(query={"file": "image.png"}))

        self.assertEqual([item["name"] for item in listing["files"]], ["image.png"])
        self.assertEqual(preview.status, 200)
        self.assertEqual(stream.status, 200)

    async def test_preview_decoding_uses_a_worker_without_disabling_pillow_limits(self):
        image_path = self.workspace / "threaded.png"
        Image.new("RGB", (3, 2), "blue").save(image_path)
        original_limit = Image.MAX_IMAGE_PIXELS
        calls = []

        async def run_in_worker(function, *args, **kwargs):
            calls.append(function)
            return function(*args, **kwargs)

        with patch.object(server_module.asyncio, "to_thread", side_effect=run_in_worker):
            response = await self.server.preview(SimpleNamespace(query={"file": "threaded.png"}))

        self.assertEqual(response.status, 200)
        self.assertEqual(calls, [server_module.render_image_preview])
        self.assertEqual(Image.MAX_IMAGE_PIXELS, original_limit)

    async def test_preview_rejects_malformed_dimensions_and_oversized_sources(self):
        image_path = self.workspace / "bounded.png"
        Image.new("RGB", (3, 2), "blue").save(image_path)

        malformed = await self.server.preview(
            SimpleNamespace(query={"file": "bounded.png", "width": "not-a-number"})
        )
        with patch.object(server_module, "MAX_PREVIEW_IMAGE_PIXELS", 4):
            oversized = await self.server.preview(SimpleNamespace(query={"file": "bounded.png"}))
        with patch("PIL.Image.open", side_effect=Image.DecompressionBombError("bomb")):
            pillow_bomb = await self.server.preview(SimpleNamespace(query={"file": "bounded.png"}))

        self.assertEqual(malformed.status, 400)
        self.assertEqual(oversized.status, 400)
        self.assertEqual(pillow_bomb.status, 400)


class SeparateDataRootUploadTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "work-volume"
        self.data = self.root / "data-volume"
        self.workspace.mkdir()
        self.data.mkdir()
        self.server = WebServer(modules={}, work_dir=str(self.workspace), data_dir=str(self.data))

    def tearDown(self):
        self.temporary.cleanup()

    async def test_upload_returns_data_root_identifier_and_every_file_route_resolves_it(self):
        encoded = BytesIO()
        Image.new("RGB", (3, 2), "green").save(encoded, format="PNG")
        upload = SimpleNamespace(filename="source.png", file=BytesIO(encoded.getvalue()))

        class UploadRequest:
            async def post(self):
                return {"file": upload, "type": "images"}

        upload_response = await self.server.filePost(UploadRequest())
        payload = json.loads(upload_response.text)
        identifier = payload["path"]

        self.assertEqual(upload_response.status, 200)
        self.assertEqual(identifier, "@data/images/source.png")
        self.assertNotIn(str(self.root), upload_response.text)
        self.assertTrue((self.data / "images" / "source.png").is_file())

        file_response = await self.server.fileGet(SimpleNamespace(query={"file": identifier}))
        preview_response = await self.server.preview(SimpleNamespace(query={"file": identifier}))
        stream_response = await self.server.stream(SimpleNamespace(query={"file": identifier}))
        probe_response = await self.server.media_probe(
            SimpleNamespace(query={"file": identifier, "media_kind": "image"})
        )

        self.assertEqual(file_response.status, 200)
        self.assertEqual(preview_response.status, 200)
        self.assertEqual(stream_response.status, 200)
        self.assertEqual(probe_response.status, 200)

    async def test_data_listing_uses_identifiers_and_traversal_is_rejected(self):
        image_path = self.data / "images" / "listed.png"
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (2, 2), "purple").save(image_path)

        listing_response = await self.server.listdir(
            SimpleNamespace(query={"path": "@data/images"})
        )
        listing = json.loads(listing_response.text)

        self.assertEqual(listing_response.status, 200)
        self.assertEqual(listing["path"], "@data/images")
        self.assertEqual(listing["abs_path"], "@data/images")
        self.assertEqual(listing["files"][0]["path"], "@data/images/listed.png")
        self.assertNotIn(str(self.root), listing_response.text)

        traversal = "@data/images/../../outside.png"
        responses = [
            await self.server.fileGet(SimpleNamespace(query={"file": traversal})),
            await self.server.preview(SimpleNamespace(query={"file": traversal})),
            await self.server.stream(SimpleNamespace(query={"file": traversal})),
            await self.server.listdir(SimpleNamespace(query={"path": traversal})),
        ]
        self.assertTrue(all(response.status == 403 for response in responses))


if __name__ == "__main__":
    unittest.main()
